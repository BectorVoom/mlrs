//! `ts_persist` (TS-PERSIST, prototype) — the `mlrs-timeseries` half of the mlrs
//! model file format: the container discriminator and the aliases `Arima` writes
//! and reads through.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## The smallest family, and the one whose file is a RESUMPTION point
//!
//! `Arima` is the only estimator here, and its fitted state is a handful of
//! short `f64` vectors — the AR and MA coefficient blocks, the differencing
//! tail, and the Kalman filter's final state and covariance. There is no matrix
//! anywhere, so these are the smallest model files mlrs writes, a few hundred
//! bytes for a typical order.
//!
//! What makes them interesting is the last two arrays. `final_state_` and
//! `final_cov_` are where the Kalman recursion STOPPED, and `forecast` continues
//! from exactly there — so an ARIMA file is not a description of a fitted model
//! but a resumption point for one, the same way
//! [`IncrementalPCA`](crate::decomposition::IncrementalPCA)'s is. A file that
//! stored only the coefficients would load, report every information criterion
//! correctly, and forecast from a zero state: right in its attributes and wrong
//! in its predictions, with nothing to signal it.
//!
//! Everything is `F64` regardless of the estimator's `F`. `Arima` holds every
//! fitted quantity as `f64` already — the Kalman pass and the Gaussian MLE both
//! lose their conditioning at `f32`, so `F` is only the width of the SERIES the
//! estimator was handed — and the file stores what the model holds.
//!
//! | name | dtype | shape |
//! |---|---|---|
//! | `arparams_` / `maparams_` | `F64` | `[p]` / `[q]` |
//! | `diff_last_` | `F64` | `[d]` |
//! | `final_state_` | `F64` | `[state_dim]` |
//! | `final_cov_` | `F64` | `[state_dim, state_dim]` |
//! | `param:p` / `param:d` / `param:q` | `__metadata__` | — |
//! | `sigma2_` / `loglik_` / `aic_` / `aicc_` / `bic_` / `nobs_` / `converged_` | `__metadata__` | — |
//!
//! `p`, `d` and `q` are stored as scalars even though the first two are implied
//! by `arparams_`'s and `diff_last_`'s lengths, because the third is NOT — a
//! `q = 0` model writes an empty `maparams_`, and an absent-vs-empty distinction
//! is exactly what a length cannot carry. Storing all three keeps the order one
//! fact rather than three inferences, and every one of them is cross-checked
//! against the arrays on load.
//!
//! Tests live in `crates/mlrs-algos/tests/ts_persist_test.rs` (AGENTS.md §2).

// The container is shared with every other family; only the discriminator is
// local. Re-exported (not just imported) so
// `timeseries::ts_persist::{AlignedBytes, SaveModel, …}` is the single import
// path for a time-series estimator's `save`/`load`.
pub use crate::persist::{
    as_f64, expect_len, shape_1d, shape_2d, AlignedBytes, Container, LoadModel, ModelFile,
    ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The time-series container discriminator (`format = "mlrs-timeseries"`).
pub struct TimeSeriesContainer;

impl Container for TimeSeriesContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-timeseries";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`TimeSeriesFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The time-series writer: [`ModelWriter`] pinned to the `mlrs-timeseries`
/// container.
pub type TimeSeriesWriter<'a> = ModelWriter<'a, TimeSeriesContainer>;

/// The time-series reader: [`ModelFile`] pinned to the `mlrs-timeseries`
/// container.
pub type TimeSeriesFile<'a> = ModelFile<'a, TimeSeriesContainer>;

/// Read a REQUIRED `f64` vector of a known length.
///
/// Every array in this family is short and its length is implied by the order,
/// so each one is measured against that rather than trusted — the file is
/// untrusted input (T-04-01-01), and `forecast` indexes the coefficient blocks
/// by lag without a bound of its own.
///
/// An EMPTY vector is legitimate here and is accepted: `p = 0` or `q = 0` is an
/// ordinary ARIMA order, so `expected == 0` means the tensor must be empty, not
/// that it must be absent.
pub fn read_f64_vec(
    file: &TimeSeriesFile<'_>,
    name: &'static str,
    expected: usize,
) -> Result<Vec<f64>, PersistError> {
    let view = file.tensor(name)?;
    expect_len(name, shape_1d(&view, name)?, expected, "entries")?;
    Ok(as_f64(&view, name)?.into_owned())
}
