//! Row gathering for Rust-native callers (MODSEL-RS-03).
//!
//! [`split`](super::split) produces *indices*; this module turns them back into
//! data. The trait exists because "the same rows out of every container the
//! caller passed" is exactly what `train_test_split` promises, and a Rust user
//! holding a `polars::DataFrame` and an `ndarray::Array2` wants one call that
//! handles both.
//!
//! ## Egress mirrors ingress
//!
//! Every implementation returns the SAME container type it was given — an
//! `Array2` in, an `Array2` out; a `DataFrame` in, a `DataFrame` out. This
//! mirrors the D-03 rule the Python shim follows and keeps a split composable
//! with whatever the caller does next.
//!
//! ## Feature gating
//!
//! `ndarray` and `polars` are **optional** dependencies, off by default:
//!
//! ```toml
//! mlrs-algos = { version = "0.1", features = ["ndarray", "polars"] }
//! ```
//!
//! Slices and `Vec`s always work with no features at all, so an embedding
//! caller that already owns its own matrix type pays for nothing. `polars` in
//! particular is a heavy build, and the mlrs wheels never need it — the Python
//! side reaches polars through polars' own Python API, not through this crate.

/// A container whose rows can be gathered by positional index.
///
/// Indexing is **positional**, matching sklearn's `_safe_indexing`: a container
/// carrying its own labels (a polars `DataFrame` has none, but a future pandas-
/// like one would) is taken by position, never by label.
pub trait RowContainer {
    /// What a gather produces — the same container type, by the egress-mirrors-
    /// ingress rule above.
    type Output;

    /// Number of rows (sklearn's `_num_samples`).
    fn n_rows(&self) -> usize;

    /// Gather `indices` in the order given.
    ///
    /// # Panics
    /// Panics on an out-of-range index, like numpy's fancy indexing. Indices
    /// produced by this crate's splitters are in range by construction.
    fn take_rows(&self, indices: &[i64]) -> Self::Output;
}

impl<T: Clone> RowContainer for [T] {
    type Output = Vec<T>;

    fn n_rows(&self) -> usize {
        self.len()
    }

    fn take_rows(&self, indices: &[i64]) -> Vec<T> {
        indices.iter().map(|&i| self[i as usize].clone()).collect()
    }
}

impl<T: Clone> RowContainer for Vec<T> {
    type Output = Vec<T>;

    fn n_rows(&self) -> usize {
        self.len()
    }

    fn take_rows(&self, indices: &[i64]) -> Vec<T> {
        self.as_slice().take_rows(indices)
    }
}

/// A row-major dense matrix held as a flat buffer plus a column count — the
/// shape every other mlrs estimator already speaks.
///
/// This is the no-dependency matrix container: `RowMajor { data, n_cols }` over
/// a `&[f64]` gathers whole rows without pulling in `ndarray`.
#[derive(Debug, Clone, Copy)]
pub struct RowMajor<'a, T> {
    pub data: &'a [T],
    pub n_cols: usize,
}

impl<T: Clone> RowContainer for RowMajor<'_, T> {
    type Output = Vec<T>;

    fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.data.len() / self.n_cols
        }
    }

    fn take_rows(&self, indices: &[i64]) -> Vec<T> {
        let mut out = Vec::with_capacity(indices.len() * self.n_cols);
        for &i in indices {
            let start = i as usize * self.n_cols;
            out.extend_from_slice(&self.data[start..start + self.n_cols]);
        }
        out
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::RowContainer;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

    impl<T: Clone> RowContainer for Array2<T> {
        type Output = Array2<T>;

        fn n_rows(&self) -> usize {
            self.nrows()
        }

        fn take_rows(&self, indices: &[i64]) -> Array2<T> {
            let idx: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            self.select(Axis(0), &idx)
        }
    }

    impl<T: Clone> RowContainer for ArrayView2<'_, T> {
        type Output = Array2<T>;

        fn n_rows(&self) -> usize {
            self.nrows()
        }

        fn take_rows(&self, indices: &[i64]) -> Array2<T> {
            let idx: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            self.select(Axis(0), &idx)
        }
    }

    impl<T: Clone> RowContainer for Array1<T> {
        type Output = Array1<T>;

        fn n_rows(&self) -> usize {
            self.len()
        }

        fn take_rows(&self, indices: &[i64]) -> Array1<T> {
            let idx: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            self.select(Axis(0), &idx)
        }
    }

    impl<T: Clone> RowContainer for ArrayView1<'_, T> {
        type Output = Array1<T>;

        fn n_rows(&self) -> usize {
            self.len()
        }

        fn take_rows(&self, indices: &[i64]) -> Array1<T> {
            let idx: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            self.select(Axis(0), &idx)
        }
    }
}

#[cfg(feature = "polars")]
mod polars_impl {
    use super::RowContainer;
    use polars::prelude::*;

    impl RowContainer for DataFrame {
        type Output = PolarsResult<DataFrame>;

        fn n_rows(&self) -> usize {
            self.height()
        }

        /// `DataFrame::take` over an `IdxCa` — polars' own gather, so the
        /// result keeps every column's dtype and the frame's schema.
        fn take_rows(&self, indices: &[i64]) -> PolarsResult<DataFrame> {
            let idx: Vec<IdxSize> = indices.iter().map(|&i| i as IdxSize).collect();
            let ca = IdxCa::from_vec(PlSmallStr::from_static("idx"), idx);
            self.take(&ca)
        }
    }

    impl RowContainer for Series {
        type Output = PolarsResult<Series>;

        fn n_rows(&self) -> usize {
            self.len()
        }

        fn take_rows(&self, indices: &[i64]) -> PolarsResult<Series> {
            let idx: Vec<IdxSize> = indices.iter().map(|&i| i as IdxSize).collect();
            let ca = IdxCa::from_vec(PlSmallStr::from_static("idx"), idx);
            self.take(&ca)
        }
    }
}

/// Gather the train and test halves of a [`Split`](super::split::Split) out of
/// one container in a single call.
///
/// ```ignore
/// let split = train_test_split_indices(x.n_rows(), SizeSpec::Float(0.25),
///                                      SizeSpec::None, true, None, &mut rng)?;
/// let (x_train, x_test) = take_split(&x, &split);
/// let (y_train, y_test) = take_split(&y, &split);
/// ```
pub fn take_split<C: RowContainer + ?Sized>(
    container: &C,
    split: &super::split::Split,
) -> (C::Output, C::Output) {
    (
        container.take_rows(&split.train),
        container.take_rows(&split.test),
    )
}
