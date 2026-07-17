//! Boundary error → `PyErr` mapping (D-10 / CLAUDE.md: `anyhow` at the binding
//! boundary, `thiserror` stays in the libraries).
//!
//! This module is the single place the `mlrs-py` cdylib translates the workspace
//! typed errors ([`mlrs_core::error::BridgeError`], [`mlrs_algos::error::AlgoError`])
//! and an opaque boundary [`anyhow::Error`] into the *right* Python exception
//! type. Keeping it in one module means every `#[pymethods]` entry maps failures
//! uniformly, and the choice of Python exception class is auditable in one place:
//!
//! | Source error                                   | Python exception |
//! |------------------------------------------------|------------------|
//! | `BridgeError::Offset` (sliced/offset array)    | `PyValueError`   |
//! | `BridgeError::HasNulls`                         | `PyValueError`   |
//! | `BridgeError::Misaligned`                       | `PyValueError`   |
//! | `BridgeError::DataTypeMismatch`                 | `PyTypeError`    |
//! | unsupported / non-float Arrow dtype            | `PyTypeError`    |
//! | f64-on-incapable-backend capability failure    | `PyValueError`   |
//! | `AlgoError` (hyperparameter / not-fitted / …)  | `PyValueError`   |
//! | `BuildError` (Phase-10 construction / enum)    | `PyValueError`   |
//! | opaque `anyhow::Error`                          | `PyRuntimeError` |
//!
//! Rationale: a malformed/aliased buffer or an out-of-range hyperparameter is a
//! *value* problem (`ValueError`); a wrong Arrow *type* (a non-float dtype, or a
//! float-kind mismatch) is a *type* problem (`TypeError`); anything we cannot
//! classify falls back to `RuntimeError` rather than silently masquerading as a
//! more specific class.

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::metrics::MetricError;
use mlrs_core::error::BridgeError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::PyErr;

/// Map a [`BridgeError`] from the Arrow→CubeCL ingress bridge to a `PyErr`.
///
/// Offset / null / misalignment violations are *value* errors (the buffer's
/// contents/shape are wrong); a `DataTypeMismatch` is a *type* error (the caller
/// handed the wrong Arrow element type). The bridge's own `Display` message is
/// preserved verbatim so the Python traceback carries the precise reason.
pub fn bridge_err_to_py(err: BridgeError) -> PyErr {
    match err {
        BridgeError::DataTypeMismatch { .. } => PyTypeError::new_err(err.to_string()),
        BridgeError::Offset { .. }
        | BridgeError::HasNulls { .. }
        | BridgeError::Misaligned { .. } => PyValueError::new_err(err.to_string()),
    }
}

/// Map an [`AlgoError`] (an estimator hyperparameter / not-fitted / unsupported /
/// convergence failure) to a `PyErr`.
///
/// These are all caller-supplied-value or usage problems, so they map to
/// `PyValueError` with the typed error's `Display` text preserved. (A `NotFitted`
/// is exposed in Python as sklearn's `NotFittedError` by the pure-Python shim;
/// the Rust boundary surfaces it as a clear `ValueError` and the shim can refine
/// it — Plan 03/04.)
pub fn algo_err_to_py(err: AlgoError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Map a [`MetricError`] (`crates/mlrs-algos/src/metrics/mod.rs`, TASK-01) to
/// a `PyErr` (TASK-15, METR-BIND-01).
///
/// `MetricError` is a DISTINCT type from [`AlgoError`] (SPEC §4 explicit
/// correction) — this is a NEW sibling of `algo_err_to_py`, not a reuse of
/// it, since `algo_err_to_py` only accepts `AlgoError`. Every `MetricError`
/// variant is a caller-supplied-value/usage problem (a length mismatch, an
/// invalid weight, an undefined single-class `roc_auc_score`, an
/// OvO+`sample_weight` combination the pinned sklearn itself rejects), so
/// all map to `PyValueError` with the typed error's `Display` text
/// preserved — the same class convention as `algo_err_to_py`/`build_err_to_py`.
pub fn metric_err_to_py(err: MetricError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Map a [`BuildError`] (a Phase-10 construction-time hyperparameter / invalid
/// enum-string failure from a builder `build()` or a `Loss`/`Penalty`/
/// `LearningRate` `TryFrom<&str>`) to a `PyErr` (D-09).
///
/// These are all data-INDEPENDENT, caller-supplied-value problems, so they map
/// to `PyValueError` with the typed error's `Display` text preserved — the same
/// class `algo_err_to_py` uses. sklearn raises these at construction; mlrs
/// surfaces them at the first `fit` (the Unfit arm stores the raw strings until
/// then). Folding the enum-parse failures into `BuildError` means this SINGLE
/// mapper covers every construction failure (mirrors the single-site
/// `algo_err_to_py` rationale).
pub fn build_err_to_py(err: BuildError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Build the canonical "estimator not fitted" `PyErr` the `#[pyclass]` wrappers
/// raise when an output method / fitted-attribute accessor is called before
/// `fit` (T-06-09).
///
/// Mirrors the algos-level [`AlgoError::NotFitted`] `Display` text so the Python
/// shim sees a uniform message and re-raises it as
/// `sklearn.exceptions.NotFittedError`. Surfaced as a `PyValueError` (the same
/// class `algo_err_to_py` uses for a `NotFitted`), so the wrapper's own
/// not-fitted path and the algos-level one are indistinguishable to the shim.
pub fn not_fitted(estimator: &str, operation: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{estimator} is not fitted yet: call `fit` before `{operation}`"
    ))
}

/// Build the `PyErr` raised when a dtype-specific accessor (e.g. the f32
/// `predict_proba` path) is called on an estimator fitted as the OTHER dtype
/// (WR-04).
///
/// The estimator IS fitted — it is simply the wrong dtype — so surfacing a
/// `not_fitted` "called before fit" error would mislead a Python user who fitted
/// in `fitted_dtype` and called the `requested_dtype` accessor. This is a caller
/// *value* problem, so it maps to `PyValueError` naming the actual fitted dtype
/// and the dtype-matched accessor to call instead.
pub fn dtype_mismatch(estimator: &str, requested_dtype: &str, fitted_dtype: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{estimator} was fitted as {fitted_dtype}; the {requested_dtype} accessor \
         does not apply — call the {fitted_dtype} accessor instead"
    ))
}

/// Build the `PyErr` raised when a `partial_fit` stream mixes float dtypes — a
/// batch's dtype disagrees with the dtype the estimator's first batch fixed.
///
/// Streaming `partial_fit` builds the fitted arm (`F32`/`F64`) from the FIRST
/// batch's dtype; a later batch of the other dtype cannot be merged into that
/// monomorphization. This is a caller *value* problem, so it maps to
/// `PyValueError` (the same class as the other ingress/usage errors).
pub fn dtype_mismatch_in_stream(estimator: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{estimator}: partial_fit batch dtype disagrees with the dtype fixed by \
         the first batch; keep every batch the same float dtype"
    ))
}

/// Map an opaque boundary [`anyhow::Error`] to a `PyErr`.
///
/// Used for the few failures that arrive as `anyhow` at the binding boundary
/// (D-10) and carry no more specific typed classification — these surface as
/// `PyRuntimeError` so they are never silently mis-typed as a `ValueError`.
pub fn anyhow_err_to_py(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

/// Build the canonical "unsupported Arrow dtype" `PyTypeError` for a non-float
/// input array.
///
/// The ingress path only accepts `Float32`/`Float64`; any other Arrow `DataType`
/// reaches this and becomes a `PyTypeError` naming the offending dtype, so the
/// Python caller gets a clear "this estimator needs a float array" message rather
/// than a downcast panic.
pub fn unsupported_dtype_err(found: &arrow::datatypes::DataType) -> PyErr {
    PyTypeError::new_err(format!(
        "mlrs: unsupported Arrow dtype {found:?}; expected Float32 or Float64"
    ))
}
