//! Arrow PyCapsule ingress (D-02 / PY-03) — owned capsule import → the *unchanged*
//! `mlrs_backend::bridge` validation → a pooled `DeviceArray`.
//!
//! This is the single host→device entry for the Python surface and the primary
//! FFI threat surface of the phase (T-06-03 / T-06-04). The ownership and
//! validation contracts are:
//!
//! 1. **Owned capsule import (no `&[u8]` borrow — PY-03 / T-06-03).**
//!    [`capsule_to_array`] consumes the Python array's `__arrow_c_array__`
//!    capsule via arrow-rs `FromPyArrow` (`ArrayData::from_pyarrow_bound`), which
//!    takes ownership of the C `ArrowArray` *including its release callback*, and
//!    produces an **owned** `ArrayRef`. Nothing borrows into the Python-owned
//!    buffer past this call, so there is no use-after-free / double-free of the
//!    Arrow C array across the boundary.
//!
//! 2. **Reuse the validate bridge UNCHANGED (D-02 / T-06-04).** [`validated_f32`]
//!    / [`validated_f64`] downcast the owned array to `Float32Array` /
//!    `Float64Array` and feed it to `mlrs_backend::bridge::validate_f32` /
//!    `validate_f64` *verbatim* — the same hard-reject validator the rest of the
//!    workspace uses. A sliced/offset array is rejected (`BridgeError::Offset`)
//!    before any upload, so aliased parent-buffer data never reaches the device.
//!
//! 3. **Single metered upload.** The validated `&[F]` is uploaded with
//!    [`mlrs_backend::device_array::DeviceArray::from_host`], which meters the
//!    allocation through the shared [`BufferPool`] and performs exactly one host
//!    copy (the honest A3 semantics carried over from Phase 1).
//!
//! The `#[pyclass]` wrappers (Plan 03) call these from inside `Python::detach`;
//! these functions are plain Rust (no `Python<'_>` capture) so they are usable in
//! the detached, GIL-released closure.

use arrow::array::{
    make_array, Array, ArrayData, ArrayRef, Float32Array, Float64Array, UInt32Array,
};
use arrow::datatypes::DataType;
use arrow::pyarrow::FromPyArrow;
use mlrs_backend::bridge::{validate_f32, validate_f64, validate_u32};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use pyo3::prelude::*;

use crate::errors::{bridge_err_to_py, unsupported_dtype_err};

/// Consume a Python pyarrow array's `__arrow_c_array__` capsule into an **owned**
/// arrow-rs [`ArrayRef`] (PY-03 / T-06-03).
///
/// arrow-rs's `FromPyArrow` takes ownership of the C `ArrowArray` and its release
/// callback; the returned [`ArrayRef`] owns its buffers. There is no `&[u8]`
/// borrow into the Python-owned buffer — the caller is free to drop the Python
/// handle once this returns.
pub fn capsule_to_array(x: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    // `from_pyarrow_bound` is the only `FromPyArrow` ingress in arrow 59 (the
    // resolved symbol from Plan 01's `arrow_symbol_probe.rs`). It returns an
    // OWNED `ArrayData`; `make_array` wraps it into an `ArrayRef`.
    let data: ArrayData = ArrayData::from_pyarrow_bound(x)?;
    Ok(make_array(data))
}

/// Validate an owned `Float32Array` via the UNCHANGED bridge and upload it to a
/// pooled [`DeviceArray`] (D-02).
///
/// `arr` must be the `Float32Array` view of an owned [`ArrayRef`] (so its buffers
/// outlive the borrowed `&[f32]` `validate_f32` returns). A sliced/offset array
/// is hard-rejected as a `PyValueError` before any upload.
pub fn validated_f32(
    arr: &Float32Array,
    pool: &mut BufferPool<ActiveRuntime>,
) -> PyResult<DeviceArray<ActiveRuntime, f32>> {
    let validated: &[f32] = validate_f32(arr).map_err(bridge_err_to_py)?;
    Ok(DeviceArray::from_host(pool, validated))
}

/// Validate an owned `Float64Array` via the UNCHANGED bridge and upload it to a
/// pooled [`DeviceArray`] (D-02).
///
/// The f64-on-incapable-backend guard ([`crate::capability::guard_f64`]) is the
/// caller's responsibility on the f64 dispatch arm (D-04) — it must run *before*
/// this upload so an f64-incapable backend never allocates the device buffer.
pub fn validated_f64(
    arr: &Float64Array,
    pool: &mut BufferPool<ActiveRuntime>,
) -> PyResult<DeviceArray<ActiveRuntime, f64>> {
    let validated: &[f64] = validate_f64(arr).map_err(bridge_err_to_py)?;
    Ok(DeviceArray::from_host(pool, validated))
}

/// Validate an owned `Float32Array` via the UNCHANGED bridge and BORROW its
/// values — the no-upload sibling of [`validated_f32`].
///
/// Same hard-reject validator, same sliced/offset rejection; the only
/// difference is that nothing is copied to the device. For a prim that consumes
/// the operand from host memory (`prims::linear_predict::linear_predict_from_host`
/// on the cpu backend, where "device" memory IS host memory) the upload is a
/// pure `memcpy` of the whole matrix and dominates the operation — see that
/// prim's docs for the measurement.
///
/// The returned slice borrows the caller's `arr`, which must itself be the view
/// of an OWNED [`ArrayRef`] (the [`capsule_to_array`] contract), so the data
/// outlives the borrow without anything reaching back into Python-owned memory
/// that could be freed underneath it.
pub fn host_slice_f32<'a>(arr: &'a Float32Array) -> PyResult<&'a [f32]> {
    validate_f32(arr).map_err(bridge_err_to_py)
}

/// f64 twin of [`host_slice_f32`]. The f64-on-incapable-backend guard
/// ([`crate::capability::guard_f64`]) remains the caller's responsibility on the
/// f64 dispatch arm (D-04), exactly as for [`validated_f64`].
pub fn host_slice_f64<'a>(arr: &'a Float64Array) -> PyResult<&'a [f64]> {
    validate_f64(arr).map_err(bridge_err_to_py)
}

/// Label twin of [`host_slice_f32`], for `VotingClassifier`'s hard-voting route
/// (VOTE-CLF-01).
///
/// The members' `predict` answers are ENCODED class labels — indices, not
/// measurements — so the column that crosses is `uint32` and there is no
/// f32/f64 dispatch to make. It runs the same `mlrs_backend::bridge` gauntlet as
/// the float ingress; a label array reaching the FFI is no more trusted than a
/// feature matrix.
pub fn host_slice_u32<'a>(arr: &'a UInt32Array) -> PyResult<&'a [u32]> {
    validate_u32(arr).map_err(bridge_err_to_py)
}

/// Is every value in `v` finite (no NaN, no ±inf)?
///
/// The host-side half of the RELOCATED NaN/inf rejection: a wrapper that passes
/// `ensure_all_finite=False` to the shim's `_normalize` has moved the check
/// here, it has NOT dropped it — pair this with
/// [`crate::errors::nonfinite_input_err`], which reproduces `check_array`'s
/// exact message so the contract is unchanged from Python.
///
/// ## Why the sum, and why it is sound
/// A plain `iter().all(f32::is_finite)` short-circuits, which stops the compiler
/// vectorizing it; on a 25 MB operand that is the difference between a
/// memory-bound pass and a scalar one. Summing instead is a branch-free
/// reduction over eight independent accumulators, and it is a valid FILTER
/// because non-finiteness propagates: a NaN input makes the sum NaN, and a lone
/// infinity makes it infinite (two opposite infinities give NaN — still not
/// finite). So a FINITE sum proves every input was finite, and the function can
/// return `true` immediately.
///
/// The converse does not hold — a sum can overflow to infinity from entirely
/// finite inputs — which is why a non-finite sum falls through to the exact
/// elementwise scan rather than reporting a violation. The fast path is a
/// filter, never the verdict. This is the same structure `numpy`'s own
/// `_assert_all_finite` uses.
pub fn all_finite_f32(v: &[f32]) -> bool {
    let mut acc = [0f32; 8];
    let mut chunks = v.chunks_exact(8);
    for c in &mut chunks {
        for i in 0..8 {
            acc[i] += c[i];
        }
    }
    let mut total: f32 = acc.iter().sum();
    for &x in chunks.remainder() {
        total += x;
    }
    total.is_finite() || v.iter().all(|x| x.is_finite())
}

/// f64 twin of [`all_finite_f32`]. Same filter-then-verify structure.
pub fn all_finite_f64(v: &[f64]) -> bool {
    let mut acc = [0f64; 8];
    let mut chunks = v.chunks_exact(8);
    for c in &mut chunks {
        for i in 0..8 {
            acc[i] += c[i];
        }
    }
    let mut total: f64 = acc.iter().sum();
    for &x in chunks.remainder() {
        total += x;
    }
    total.is_finite() || v.iter().all(|x| x.is_finite())
}

/// Downcast an owned [`ArrayRef`] to a `Float32Array`, or a `PyTypeError` if the
/// array is not Float32.
///
/// Used by the dtype-dispatch (D-06): once the dtype is known to be Float32 the
/// wrapper takes the concrete view to feed [`validated_f32`].
pub fn as_f32(array: &ArrayRef) -> PyResult<&Float32Array> {
    array
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| unsupported_dtype_err(array.data_type()))
}

/// Downcast an owned [`ArrayRef`] to a `Float64Array`, or a `PyTypeError` if the
/// array is not Float64.
pub fn as_f64(array: &ArrayRef) -> PyResult<&Float64Array> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| unsupported_dtype_err(array.data_type()))
}

/// Downcast an owned [`ArrayRef`] to a `UInt32Array`, or a `PyTypeError` if the
/// array is not UInt32.
///
/// The shim promotes its label columns to `uint32` before the call rather than
/// sending whatever integer width the members happened to return, so this
/// downcast has a single accepted dtype rather than an integer dispatch — see
/// `_vote_labels_via_rust`.
pub fn as_u32(array: &ArrayRef) -> PyResult<&UInt32Array> {
    array
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| unsupported_dtype_err(array.data_type()))
}

/// The float-kind of an owned ingress array, for the D-06 dtype dispatch.
///
/// Returns `Ok(FloatDtype::F32 | F64)` for the two supported element types, or a
/// `PyTypeError` naming the unsupported dtype for anything else (so a non-float
/// Arrow array never reaches a downcast panic).
pub fn float_dtype(array: &ArrayRef) -> PyResult<FloatDtype> {
    match array.data_type() {
        DataType::Float32 => Ok(FloatDtype::F32),
        DataType::Float64 => Ok(FloatDtype::F64),
        other => Err(unsupported_dtype_err(other)),
    }
}

/// The two host-float element kinds the ingress accepts (D-06 dispatch key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDtype {
    /// 32-bit float (`Float32Array`).
    F32,
    /// 64-bit float (`Float64Array`).
    F64,
}
