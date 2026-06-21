//! The 12 `#[pyclass]` estimator wrappers (PY-01/PY-02/PY-05) — the compiled half
//! of the Python surface the pure-Python `mlrs` shim (Plan 04) subclasses and
//! delegates to.
//!
//! Each wrapper is a `#[pyclass]` holding an internal `Any<Name>` dtype-dispatch
//! enum (emitted by the [`any_estimator!`](crate::any_estimator) macro, Plan 02):
//! an `Unfit { .. }` arm storing the verbatim sklearn-named hyperparameters
//! (PY-02), plus the two fitted monomorphizations `F32(Estimator<f32>)` /
//! `F64(Estimator<f64>)` (D-06).
//!
//! Every device-compute `#[pymethods]` body honors the two load-bearing contracts
//! documented on [`crate::dispatch`]:
//!
//! 1. **GIL release (PY-03).** The trait call runs inside `py.detach(|| { … })`
//!    around a lock of the process-global pool ([`crate::global_pool`]).
//! 2. **f64 guard (D-04).** On the `FloatDtype::F64` dispatch arm,
//!    [`crate::capability::guard_f64`]`()?` runs BEFORE any upload, so f64 on an
//!    f64-incapable backend raises a clear `PyValueError` and never allocates a
//!    device buffer or silently downcasts.
//!
//! Fitted-attribute accessors materialize host buffers (`Vec<f32>`/`Vec<f64>` for
//! floats, `Vec<i32>` for labels/indices — D-03/D-06) for the shim, which wraps
//! them to the resolved `output_type`. Accessing a fitted attribute (or calling an
//! output method) before `fit` returns the algos-level `NotFitted` error, which
//! [`crate::errors`] maps to a `PyValueError` the shim re-raises as
//! `sklearn.exceptions.NotFittedError`.
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

pub mod cluster;
pub mod covariance;
pub mod decomposition;
pub mod kernel;
pub mod linear;
pub mod neighbors;
pub mod projection;
