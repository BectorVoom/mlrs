//! `frame` — polars `DataFrame` / `Series` ingress and egress (FSEL-01).
//!
//! FEATURE-GATED behind `polars`, OFF by default. polars is a very large
//! dependency tree, and an mlrs user working with `&[f32]` slices should not pay
//! for it — so the gate is not caution, it is the whole reason this compiles as
//! an optional module rather than living in `mlrs-backend`'s always-on ingress.
//!
//! ## What "Rust supports polars and numpy" means concretely
//! mlrs's Rust estimator surface takes a FLAT ROW-MAJOR slice plus an explicit
//! `(rows, cols)` geometry (D-08). That IS the numpy layout: a C-contiguous
//! `float64` array's buffer is exactly such a slice, so a caller with numpy data
//! (through `numpy` crate, PyO3, `.npy` via `npyz`, or an FFI pointer) already
//! has the supported form and needs nothing from this module.
//!
//! polars is the case that genuinely needs conversion, because a `DataFrame` is
//! COLUMN-major and per-column typed, which is neither of those things. This
//! module is that conversion, in both directions:
//!
//! * [`dataframe_to_rowmajor`] — every numeric column cast to the target float
//!   and interleaved into one row-major buffer, plus the column names;
//! * [`series_to_vec`] — a single `Series` (a target vector) to a flat `Vec`;
//! * [`rowmajor_to_dataframe`] — the inverse, for returning a transformed matrix
//!   as a frame;
//! * [`take_columns`] — a boolean support mask applied to a frame, which is what
//!   a feature selector's `transform` is for a polars caller: it keeps the frame's
//!   own dtypes and NAMES instead of flattening everything to one float.
//!
//! ## Why the conversion is explicit and not a trait impl
//! There is no `From<DataFrame> for DeviceArray`: the conversion is FALLIBLE (a
//! non-numeric column, a null, a zero-column frame) and it needs a target float
//! type the frame does not carry. A `TryFrom` could express the first but not the
//! second without an inference-defeating type parameter, so a named function that
//! returns `(values, rows, cols, names)` is both clearer at the call site and
//! honest about the cost — this MATERIALISES a transposed copy, which a caller
//! with a 10-column frame should be able to see they are paying for.
//!
//! ## Nulls are REJECTED, not filled
//! A null is not a number, and silently substituting `0.0` or `NaN` for one would
//! produce a confidently wrong score. [`FrameError::NullValues`] names the column.
//! The one exception is deliberate: `f64::NAN` values that are already in the
//! column pass through, because `VarianceThreshold` is documented to accept them
//! (sklearn validates it with `ensure_all_finite="allow-nan"`). A polars null and
//! a float NaN are different things and are treated differently.
//!
//! Tests live in `crates/mlrs-core/tests/frame_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use polars::prelude::*;

use crate::float_cast::f64_to_host;

/// Why a polars conversion failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// A column's dtype cannot be cast to a float (a string, a list, a struct).
    ///
    /// Names the column AND its dtype, because "column 3 is a String" is
    /// actionable and "conversion failed" is not.
    #[error("column '{column}' has dtype {dtype} which is not numeric")]
    NonNumericColumn {
        /// The offending column's name.
        column: String,
        /// Its polars dtype, as displayed.
        dtype: String,
    },

    /// A column contains polars NULLs.
    ///
    /// Distinct from a float `NaN`, which is allowed through — see the module
    /// docs. Carries the count so a caller can tell one stray null from a mostly
    /// empty column.
    #[error("column '{column}' contains {count} null value(s); mlrs requires dense input")]
    NullValues {
        /// The offending column's name.
        column: String,
        /// How many nulls it holds.
        count: usize,
    },

    /// The frame has no columns, or no rows.
    #[error("frame has {rows} rows and {cols} columns; both must be non-zero")]
    Empty {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },

    /// A geometry disagreement building a frame from a flat buffer.
    #[error("cannot build a {rows}x{cols} frame from {len} values")]
    ShapeMismatch {
        /// Intended row count.
        rows: usize,
        /// Intended column count.
        cols: usize,
        /// Values actually supplied.
        len: usize,
    },

    /// polars itself failed (a cast that its own kernels rejected).
    ///
    /// The message is polars' own, kept verbatim rather than paraphrased.
    #[error("polars error: {0}")]
    Polars(String),
}

/// A polars `Series` cast to `f64`, or the reason it cannot be.
fn column_as_f64(name: &str, series: &Series) -> Result<Vec<f64>, FrameError> {
    if !series.dtype().is_primitive_numeric() && !matches!(series.dtype(), DataType::Boolean) {
        return Err(FrameError::NonNumericColumn {
            column: name.to_string(),
            dtype: format!("{}", series.dtype()),
        });
    }
    let nulls = series.null_count();
    if nulls > 0 {
        return Err(FrameError::NullValues {
            column: name.to_string(),
            count: nulls,
        });
    }
    let cast = series
        .cast(&DataType::Float64)
        .map_err(|e| FrameError::Polars(e.to_string()))?;
    let ca = cast.f64().map_err(|e| FrameError::Polars(e.to_string()))?;
    // `null_count == 0` was checked above, so every slot is present and the
    // `expect` cannot fire; it is an assertion about that invariant rather than a
    // hope.
    Ok(ca
        .iter()
        .map(|v| v.expect("null-free column yields no None after cast"))
        .collect())
}

/// A polars `DataFrame` as a ROW-MAJOR `Vec<F>` plus its geometry and column
/// names — the form every mlrs estimator takes (D-08).
///
/// Every column is cast to `f64` first and then narrowed to `F`, so a frame
/// mixing `i32`, `u8`, `Boolean` and `f32` columns converts consistently rather
/// than per-column. Booleans become `0.0`/`1.0`, which is what `numpy`'s own
/// `astype(float)` and sklearn's `check_array` do with them.
///
/// Returns `(values, rows, cols, names)`. The names are returned rather than
/// discarded because a selector's caller needs them for
/// `get_feature_names_out()`, and re-reading them from the frame afterwards would
/// invite the two to be taken from different frames.
pub fn dataframe_to_rowmajor<F>(
    df: &DataFrame,
) -> Result<(Vec<F>, usize, usize, Vec<String>), FrameError>
where
    F: Pod,
{
    let (rows, cols) = df.shape();
    if rows == 0 || cols == 0 {
        return Err(FrameError::Empty { rows, cols });
    }
    let names = column_names(df);

    // Convert column by column (each is contiguous in the frame), then
    // INTERLEAVE. The transpose is the unavoidable cost of a column-major source
    // feeding a row-major consumer; doing it in one pass over a pre-sized buffer
    // is what keeps it a single copy rather than `cols` reallocations.
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(cols);
    for (name, column) in names.iter().zip(df.columns()) {
        let series = column.as_materialized_series();
        columns.push(column_as_f64(name, series)?);
    }
    let mut values: Vec<F> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            values.push(f64_to_host::<F>(columns[c][r]));
        }
    }
    Ok((values, rows, cols, names))
}

/// A single polars `Series` (a target vector) as a flat `Vec<F>`.
pub fn series_to_vec<F>(series: &Series) -> Result<Vec<F>, FrameError>
where
    F: Pod,
{
    let name = series.name().to_string();
    Ok(column_as_f64(&name, series)?
        .into_iter()
        .map(f64_to_host::<F>)
        .collect())
}

/// The inverse of [`dataframe_to_rowmajor`]: a row-major buffer back into a
/// `DataFrame` with the given column names.
///
/// Every column comes back as `Float64` — the buffer carries one float type, so
/// there is no per-column dtype left to restore. A caller who needs the ORIGINAL
/// dtypes preserved should use [`take_columns`] instead, which never leaves the
/// frame domain and therefore never loses them. That distinction is the reason
/// both functions exist.
pub fn rowmajor_to_dataframe(
    values: &[f64],
    rows: usize,
    names: &[String],
) -> Result<DataFrame, FrameError> {
    let cols = names.len();
    if rows == 0 || cols == 0 {
        return Err(FrameError::Empty { rows, cols });
    }
    if values.len() != rows * cols {
        return Err(FrameError::ShapeMismatch {
            rows,
            cols,
            len: values.len(),
        });
    }
    let columns: Vec<Column> = names
        .iter()
        .enumerate()
        .map(|(c, name)| {
            let col: Vec<f64> = (0..rows).map(|r| values[r * cols + c]).collect();
            Column::new(name.as_str().into(), col)
        })
        .collect();
    // polars 0.54's `DataFrame::new` takes the HEIGHT explicitly (a zero-column
    // frame has no column to infer it from), so it is passed rather than derived.
    DataFrame::new(rows, columns).map_err(|e| FrameError::Polars(e.to_string()))
}

/// Apply a feature selector's support MASK to a `DataFrame`, keeping the frame's
/// own dtypes and column names.
///
/// This is a selector's `transform` for a polars caller, and it deliberately does
/// NOT go through [`dataframe_to_rowmajor`]: the selection is a choice of
/// COLUMNS, so performing it in the frame domain keeps every per-column dtype and
/// name intact and copies only the columns that survive. Routing it through a flat
/// `f64` buffer would flatten an `i32` column to a float, drop the names, and copy
/// the columns that are about to be discarded.
///
/// An ALL-FALSE mask yields a zero-column frame with the original row count, which
/// is sklearn's behaviour (it warns rather than raising).
pub fn take_columns(df: &DataFrame, mask: &[bool]) -> Result<DataFrame, FrameError> {
    let (rows, cols) = df.shape();
    if mask.len() != cols {
        return Err(FrameError::ShapeMismatch {
            rows,
            cols,
            len: mask.len(),
        });
    }
    let keep: Vec<PlSmallStr> = column_names(df)
        .iter()
        .zip(mask)
        .filter(|(_, &k)| k)
        .map(|(n, _)| PlSmallStr::from_str(n))
        .collect();
    df.select(keep)
        .map_err(|e| FrameError::Polars(e.to_string()))
}

/// A `DataFrame`'s column names, for `feature_names_in_` / `get_feature_names_out`
/// without a full conversion.
pub fn column_names(df: &DataFrame) -> Vec<String> {
    df.columns().iter().map(|c| c.name().to_string()).collect()
}
