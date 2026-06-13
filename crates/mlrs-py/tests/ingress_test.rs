//! Ingress + capability-guard + boundary-error integration tests (Task 06-02-1).
//!
//! AGENTS.md §2: tests live here, never in an in-source `#[cfg(test)]` module.
//!
//! These exercise the Rust side of the Arrow PyCapsule ingress path — owned-array
//! validation via the UNCHANGED `mlrs_backend::bridge`, the dtype dispatch
//! (D-06), the f64-on-incapable-backend guard (D-04), and the boundary
//! error→PyErr mapping — by constructing the same owned arrow-rs arrays that
//! `capsule_to_array` produces.
//!
//! ## Why these tests do not touch the CPython interpreter
//! `mlrs-py` builds with PyO3's `extension-module` feature (so the wheel does NOT
//! link libpython — the interpreter provides the symbols at import time). A Rust
//! integration-test binary therefore cannot link a `Python::attach` /
//! `PyErr::is_instance_of` call (the libpython symbols are undefined at link).
//! So these tests assert the *control flow* (`Ok`/`Err`) of the ingress path and
//! the *typed* `From`-mappings at the `BridgeError`/`AlgoError` layer (pure
//! Rust). The concrete Python exception **class** for each path
//! (`PyValueError` / `PyTypeError`) is asserted end-to-end by the Python pytest
//! oracle (Plan 05/06), where a live interpreter is present.
//!
//! Built only with a backend feature active (the crate is cfg-gated over
//! backends); run as `cargo test -p mlrs-py --features cpu --test ingress_test`.

#![cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array};
use arrow::buffer::ScalarBuffer;

use mlrs_core::error::BridgeError;

use mlrs_py::capability::{guard_f64, supports_f64};
use mlrs_py::ingress::{as_f32, as_f64, float_dtype, validated_f32, validated_f64, FloatDtype};

use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::active_client;

/// A fresh pool over the active-runtime client (cpu in this gate).
fn fresh_pool() -> BufferPool<mlrs_backend::runtime::ActiveRuntime> {
    BufferPool::new(active_client())
}

/// A freshly-contiguous (non-sliced) `Float32Array` — the shape the D-02 shim
/// hands across the boundary.
fn contiguous_f32(values: &[f32]) -> Float32Array {
    Float32Array::from(values.to_vec())
}

/// A freshly-contiguous (non-sliced) `Float64Array`.
fn contiguous_f64(values: &[f64]) -> Float64Array {
    Float64Array::from(values.to_vec())
}

#[test]
fn contiguous_f32_validates_and_uploads() {
    let arr = contiguous_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mut pool = fresh_pool();
    let dev = match validated_f32(&arr, &mut pool) {
        Ok(d) => d,
        Err(_) => panic!("contiguous f32 must validate"),
    };
    assert_eq!(dev.len(), 4, "device array carries the element count");
    let host = dev.to_host(&pool);
    assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0], "round-trips through the device");
}

#[test]
fn contiguous_f64_validates_and_uploads() {
    // f64 round-trip only on an f64-capable backend (cpu is); otherwise the
    // guard path is covered by `f64_guard_matches_backend_capability` below.
    if !supports_f64() {
        eprintln!("skipping f64 ingress: backend lacks f64");
        return;
    }
    let arr = contiguous_f64(&[10.0, 20.0, 30.0]);
    let mut pool = fresh_pool();
    let dev = match validated_f64(&arr, &mut pool) {
        Ok(d) => d,
        Err(_) => panic!("contiguous f64 must validate"),
    };
    assert_eq!(dev.len(), 3);
    assert_eq!(dev.to_host(&pool), vec![10.0, 20.0, 30.0]);
}

#[test]
fn sliced_f32_is_hard_rejected_before_upload() {
    // A sliced array's values view does NOT cover the whole backing buffer, so
    // the UNCHANGED bridge `validate_no_offset` rejects it (T-06-04). It must
    // surface an Err (mapped PyValueError) and never upload.
    let full = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let sliced = full.slice(1, 3); // logical [2,3,4] — a view into a larger buffer
    let mut pool = fresh_pool();
    assert!(
        validated_f32(&sliced, &mut pool).is_err(),
        "a sliced array must be hard-rejected, not uploaded"
    );
    // The underlying bridge classifies this as an Offset violation (the source
    // of the PyValueError mapping) — assert that directly at the typed layer.
    assert!(
        matches!(
            mlrs_backend::bridge::validate_f32(&sliced),
            Err(BridgeError::Offset { .. })
        ),
        "the bridge rejects the slice as BridgeError::Offset"
    );
}

#[test]
fn from_start_slice_is_rebased_and_accepted() {
    // arrow 59's `slice(0, n)` REBASES the `ScalarBuffer` so its `values()` view
    // covers exactly its own (fresh, non-aliasing) buffer — `inner.len() ==
    // values.len() * elem` and `ptr_offset() == 0`. So a from-the-start slice is
    // genuinely contiguous and is correctly ACCEPTED by the bridge (it carries
    // no aliased parent data). This documents the boundary: only a non-zero
    // offset slice (`sliced_f32_is_hard_rejected_before_upload`) aliases and is
    // rejected.
    let parent = ScalarBuffer::<f32>::from(vec![9.0f32, 8.0, 7.0, 6.0]);
    let full = Float32Array::new(parent, None);
    let head = full.slice(0, 2); // [9,8] — arrow rebases to a 2-element buffer
    let mut pool = fresh_pool();
    let dev = match validated_f32(&head, &mut pool) {
        Ok(d) => d,
        Err(_) => panic!("a rebased from-start slice is contiguous and accepted"),
    };
    assert_eq!(dev.to_host(&pool), vec![9.0, 8.0]);
}

#[test]
fn non_float_array_reports_unsupported_dtype() {
    // An Int32 array is not a supported ingress dtype: float_dtype + the
    // downcast helpers must return Err (mapped PyTypeError), never panic on the
    // downcast.
    let int_arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    assert!(float_dtype(&int_arr).is_err(), "int32 is not a float dtype");
    assert!(as_f32(&int_arr).is_err(), "int32 cannot downcast to Float32Array");
    assert!(as_f64(&int_arr).is_err(), "int32 cannot downcast to Float64Array");
}

#[test]
fn float_dtype_dispatch_picks_the_right_arm() {
    let f32_arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0f32]));
    let f64_arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0f64]));
    assert_eq!(float_dtype(&f32_arr).ok(), Some(FloatDtype::F32));
    assert_eq!(float_dtype(&f64_arr).ok(), Some(FloatDtype::F64));
    // The concrete-view downcasts succeed for the matching dtype…
    assert!(as_f32(&f32_arr).is_ok());
    assert!(as_f64(&f64_arr).is_ok());
    // …and fail for the mismatched dtype.
    assert!(as_f64(&f32_arr).is_err());
    assert!(as_f32(&f64_arr).is_err());
}

#[test]
fn f64_guard_matches_backend_capability() {
    // D-04: `guard_f64` is Ok on an f64-capable backend and Err on an incapable
    // one; its verdict must agree with `supports_f64()`. (The concrete
    // PyValueError class + the "does not support float64 / mlrs-cpu" message are
    // asserted by the Python oracle, which has a live interpreter; the message
    // text itself is unit-checked below at the source.)
    assert_eq!(
        guard_f64().is_ok(),
        supports_f64(),
        "the f64 guard verdict tracks the backend capability"
    );
}
