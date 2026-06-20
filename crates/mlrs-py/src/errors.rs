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
//! | opaque `anyhow::Error`                          | `PyRuntimeError` |
//!
//! Rationale: a malformed/aliased buffer or an out-of-range hyperparameter is a
//! *value* problem (`ValueError`); a wrong Arrow *type* (a non-float dtype, or a
//! float-kind mismatch) is a *type* problem (`TypeError`); anything we cannot
//! classify falls back to `RuntimeError` rather than silently masquerading as a
//! more specific class.

use mlrs_algos::error::AlgoError;
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
