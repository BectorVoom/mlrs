//! Negative + positive tests for the Arrow→CubeCL hard-reject bridge (D-06 /
//! FOUND-06 / Criterion 3 / ASVS V5).
//!
//! The bridge is the phase's primary threat surface: untrusted/malformed Arrow
//! buffers (non-zero offset/slice, set null bits, misaligned backing buffer)
//! MUST return a typed `mlrs_core::error::BridgeError` BEFORE any `unsafe`
//! transmute. These tests assert the *specific* error variant for each
//! malformed case (T-03-01/02/03) and that a conforming array validates to a
//! usable slice.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `mod tests` in `src/`.

use arrow::array::{Float32Array, Float64Array};
use mlrs_core::error::BridgeError;
use mlrs_backend::bridge::{self, validate_f32, validate_f64};

// ----------------------------------------------------------------------------
// Happy path: conforming arrays validate to a usable slice of correct length.
// ----------------------------------------------------------------------------

#[test]
fn validate_f32_accepts_conforming_array() {
    let arr = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]);
    let slice = validate_f32(&arr).expect("conforming f32 array must validate");
    assert_eq!(slice, &[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(slice.len(), 4);
}

#[test]
fn validate_f64_accepts_conforming_array() {
    let arr = Float64Array::from(vec![1.0f64, 2.0, 3.0]);
    let slice = validate_f64(&arr).expect("conforming f64 array must validate");
    assert_eq!(slice, &[1.0f64, 2.0, 3.0]);
    assert_eq!(slice.len(), 3);
}

// ----------------------------------------------------------------------------
// T-03-02: sliced/offset array → Err(Offset), NOT a silent upload of aliased
// parent-buffer data.
// ----------------------------------------------------------------------------

#[test]
fn validate_f64_rejects_sliced_array() {
    let n = 5usize;
    let full = Float64Array::from(vec![1.0f64; n]);
    let sliced = full.slice(1, n - 1); // logical offset == 1
    let err = validate_f64(&sliced).expect_err("sliced array must be rejected");
    match err {
        BridgeError::Offset { offset } => assert_eq!(offset, 1),
        other => panic!("expected BridgeError::Offset, got {other:?}"),
    }
}

#[test]
fn validate_f32_rejects_sliced_array() {
    let full = Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]);
    let sliced = full.slice(2, 2);
    let err = validate_f32(&sliced).expect_err("sliced array must be rejected");
    assert!(matches!(err, BridgeError::Offset { offset: 2 }));
}

// ----------------------------------------------------------------------------
// T-03-03: nullable array with set null bits → Err(HasNulls), NOT an upload of
// meaningless null-slot values.
// ----------------------------------------------------------------------------

#[test]
fn validate_f64_rejects_nullable_array() {
    let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0)]);
    let err = validate_f64(&arr).expect_err("array with nulls must be rejected");
    match err {
        BridgeError::HasNulls { null_count } => assert_eq!(null_count, 1),
        other => panic!("expected BridgeError::HasNulls, got {other:?}"),
    }
}

#[test]
fn validate_f32_rejects_nullable_array() {
    let arr = Float32Array::from(vec![Some(1.0f32), None, None, Some(4.0)]);
    let err = validate_f32(&arr).expect_err("array with nulls must be rejected");
    assert!(matches!(err, BridgeError::HasNulls { null_count: 2 }));
}

// ----------------------------------------------------------------------------
// T-03-01: misaligned backing buffer → Err(Misaligned), NOT a panic, NOT UB.
//
// A correctly-constructed arrow array is always element-aligned, so the
// misalignment path is exercised through the public `cast_validated` helper
// that the validators call internally — fed a deliberately misaligned &[u8]
// view (start offset 1 of a u8 buffer is not 4/8-byte aligned). This proves the
// reject-before-unsafe contract for the alignment class (A7: try_cast_slice
// returns a recoverable Err, never panics).
// ----------------------------------------------------------------------------

#[test]
fn cast_validated_rejects_misaligned_f32() {
    let raw = [0u8; 32];
    // Start at offset 1 → guaranteed not 4-byte aligned for f32, and length 12
    // (3 * 4 bytes) so size is divisible; only alignment is wrong.
    let misaligned = &raw[1..13];
    let err = bridge::cast_validated::<f32>(misaligned)
        .expect_err("misaligned f32 cast must be a recoverable Err, not a panic");
    assert!(matches!(err, BridgeError::Misaligned { .. }));
}

#[test]
fn cast_validated_rejects_misaligned_f64() {
    let raw = [0u8; 64];
    // Offset 1 is not 8-byte aligned for f64; length 16 (2 * 8) is size-clean.
    let misaligned = &raw[1..17];
    let err = bridge::cast_validated::<f64>(misaligned)
        .expect_err("misaligned f64 cast must be a recoverable Err, not a panic");
    assert!(matches!(err, BridgeError::Misaligned { .. }));
}

#[test]
fn cast_validated_accepts_aligned_buffer() {
    // An aligned f32 slice cast to its own bytes and back must succeed.
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let back = bridge::cast_validated::<f32>(bytes).expect("aligned cast must succeed");
    assert_eq!(back, &data);
}
