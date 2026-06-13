//! arrow-59 `FromPyArrow` symbol probe — DOCUMENTATION ONLY (Plan 02 input).
//!
//! This file is NOT compiled (it is intentionally absent from `lib.rs`'s `mod`
//! list — see RESEARCH 06 Open Question Q2). It records, for Plan 02's owned
//! Arrow-capsule ingress, the EXACT method name that arrow-59 exposes on the
//! `FromPyArrow` trait under the `pyarrow` feature.
//!
//! ## RESOLVED (Open Question Q2 / Assumption A3)
//!
//! arrow-59 exposes exactly ONE ingress method on the trait:
//!
//! ```text
//! pub trait FromPyArrow: Sized {
//!     fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self>;
//! }
//! ```
//!
//! There is NO non-`_bound` `from_pyarrow` variant in arrow-59. The trait is
//! re-exported as `arrow::pyarrow::FromPyArrow` (arrow-59 `lib.rs`:
//! `#[cfg(feature = "pyarrow")] pub use arrow_pyarrow as pyarrow;`), and
//! `impl FromPyArrow for ArrayData` is present.
//!
//! Verified against the vendored source at
//! `~/.cargo/registry/src/.../arrow-pyarrow-59.0.0/src/lib.rs:95-99` (trait)
//! and `:260-261` (`impl FromPyArrow for ArrayData`) during 06-01 Task 5.
//!
//! ## Confirmed call shape for Plan 02 (D-02 ingress)
//!
//! ```ignore
//! use arrow::array::{ArrayData, make_array};
//! use arrow::pyarrow::FromPyArrow;        // trait must be in scope
//! use pyo3::prelude::*;
//!
//! // `x: &Bound<'_, PyAny>` is the pyarrow array passed from Python.
//! let data: ArrayData = ArrayData::from_pyarrow_bound(x)?;  // owned, no &[u8] borrow
//! let array = make_array(data);                             // ArrayRef
//! ```
//!
//! Companion egress traits also confirmed present for Plan 02/03:
//!   - `ToPyArrow::to_pyarrow(&self, py) -> PyResult<Bound<PyAny>>`
//!   - `IntoPyArrow::into_pyarrow(self, py) -> PyResult<Bound<PyAny>>`
//!
//! SYMBOL: arrow::pyarrow::FromPyArrow::from_pyarrow_bound
