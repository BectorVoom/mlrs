//! `RowContainer` gate (MODSEL-RS-03).
//!
//! Run with the optional adapters enabled:
//!
//! ```text
//! cargo test -p mlrs-algos --features cpu,ndarray,polars --test model_selection_container_test
//! ```
//!
//! The `ndarray` and `polars` cases are `#[cfg]`-gated rather than
//! `#[ignore]`d, so a default-feature run compiles and passes the slice cases
//! instead of silently reporting "0 tests" for the whole file.

use mlrs_algos::model_selection::container::{take_split, RowContainer, RowMajor};
use mlrs_algos::model_selection::split::Split;

fn split() -> Split {
    Split {
        train: vec![3, 0, 2],
        test: vec![1],
    }
}

#[test]
fn slices_gather_in_the_requested_order() {
    // The ORDER matters: a `ShuffleSplit` hands back permuted indices, and a
    // gather that sorted them would break the row correspondence between two
    // containers split with the same indices.
    let xs = vec![10, 20, 30, 40];
    let (train, test) = take_split(&xs, &split());
    assert_eq!(train, vec![40, 10, 30]);
    assert_eq!(test, vec![20]);
}

#[test]
fn row_major_gathers_whole_rows() {
    let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let matrix = RowMajor {
        data: &data,
        n_cols: 3,
    };
    assert_eq!(matrix.n_rows(), 4);
    let (train, test) = take_split(&matrix, &split());
    assert_eq!(train, vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0]);
    assert_eq!(test, vec![3.0, 4.0, 5.0]);
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_gathers_rows_and_stays_an_ndarray() {
    use ndarray::{array, Array2};

    let x: Array2<f64> = array![[0., 1.], [2., 3.], [4., 5.], [6., 7.]];
    let (train, test) = take_split(&x, &split());
    assert_eq!(train, array![[6., 7.], [0., 1.], [4., 5.]]);
    assert_eq!(test, array![[2., 3.]]);

    let y = ndarray::arr1(&[0, 1, 2, 3]);
    let (y_train, y_test) = take_split(&y, &split());
    assert_eq!(y_train, ndarray::arr1(&[3, 0, 2]));
    assert_eq!(y_test, ndarray::arr1(&[1]));
}

#[cfg(feature = "polars")]
#[test]
fn polars_gathers_rows_and_stays_a_dataframe() {
    use polars::prelude::*;

    let df = df![
        "a" => [0i64, 1, 2, 3],
        "b" => ["w", "x", "y", "z"],
    ]
    .expect("valid frame");

    let (train, test) = take_split(&df, &split());
    let train = train.expect("gather succeeds");
    let test = test.expect("gather succeeds");

    assert_eq!(train.height(), 3);
    assert_eq!(
        train
            .column("a")
            .expect("column a")
            .i64()
            .expect("i64")
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![3, 0, 2]
    );
    // The schema survives the gather — a `DataFrame` in is a `DataFrame` out,
    // with its string column still a string column.
    assert_eq!(train.get_column_names(), vec!["a", "b"]);
    assert_eq!(test.height(), 1);
}
