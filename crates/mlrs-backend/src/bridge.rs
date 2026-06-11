//! Arrow→CubeCL ingress bridge with HARD-REJECT validation (FOUND-06 / D-06).
//!
//! This module is the **single ingress** for host data into device buffers and
//! the primary threat surface of Phase 01. Apache Arrow buffers are treated as
//! **untrusted input**: a sliced/offset array, a nullable array with set null
//! bits, or a misaligned backing buffer must be rejected with a typed
//! [`mlrs_core::error::BridgeError`] (D-07) **before** any `unsafe` transmute.
//! There is no "best effort" path — malformed input is a hard error, never a
//! silent upload of aliased parent-buffer data or meaningless null-slot values.
//!
//! ## Validation order (every check precedes any `unsafe`)
//! 1. `offset() != 0`            → [`BridgeError::Offset`]   (T-03-02: aliased parent data)
//! 2. `null_count()/nulls()`     → [`BridgeError::HasNulls`] (T-03-03: meaningless null slots)
//! 3. `bytemuck::try_cast_slice` → [`BridgeError::Misaligned`] (T-03-01: alignment/size — A7)
//!
//! Only after all three pass is a `&[F]` returned. The bytes-level cast uses
//! `bytemuck::try_cast_slice`, which (A7, proven in `spike_test.rs`) returns a
//! **recoverable `Err`** on alignment/size violation — it does NOT panic — so no
//! manual `ptr % align_of::<T>()` check is needed.
//!
//! ## Honest upload semantics (A3 — NOT literal host zero-copy)
//! cubecl 0.10's `cubecl::bytes::Bytes` constructors both **own** their
//! allocation; there is no borrow/no-copy constructor taking an existing
//! `&[u8]`. So [`upload`] performs exactly **one** host copy into the device
//! buffer — "validated single-upload", not literal zero-copy. We deliberately do
//! not overclaim: the guarantee is *no extra host copies beyond the single
//! upload*, plus full pre-transmute validation.

use std::mem::size_of;

use bytemuck::Pod;
use mlrs_core::error::BridgeError;

use arrow::array::{Array, ArrowPrimitiveType, Float32Array, Float64Array, PrimitiveArray};
use arrow::datatypes::{Float32Type, Float64Type};

/// Validate a [`Float32Array`] as untrusted input and return its values as
/// `&[f32]`, or a typed [`BridgeError`] (offset / nulls / misalignment) — all
/// rejection paths return BEFORE any `unsafe` transmute (D-06).
pub fn validate_f32(arr: &Float32Array) -> Result<&[f32], BridgeError> {
    validate_primitive::<Float32Type>(arr)
}

/// Validate a [`Float64Array`] as untrusted input and return its values as
/// `&[f64]`, or a typed [`BridgeError`] — all rejection paths return BEFORE any
/// `unsafe` transmute (D-06).
pub fn validate_f64(arr: &Float64Array) -> Result<&[f64], BridgeError> {
    validate_primitive::<Float64Type>(arr)
}

/// Generic hard-reject validator shared by [`validate_f32`] / [`validate_f64`].
///
/// Runs all three checks (offset → nulls → alignment) BEFORE returning a `&[T]`.
/// No `unsafe` is reachable until every check has passed.
fn validate_primitive<T>(arr: &PrimitiveArray<T>) -> Result<&[T::Native], BridgeError>
where
    T: ArrowPrimitiveType,
    T::Native: Pod,
{
    validate_no_offset(arr)?;
    validate_no_nulls(arr)?;
    // `values()` is an O(1) view into the backing buffer (no copy). The cast
    // below re-checks alignment/size for the same bytes (defense in depth, and
    // it is the path the misalignment test exercises directly via
    // `cast_validated`).
    let slice: &[T::Native] = arr.values();
    cast_validated::<T::Native>(bytemuck::cast_slice(slice))
}

/// Reject a sliced/offset array before any transmute.
///
/// In arrow 59 `PrimitiveArray::offset()` always returns 0 (slicing rebases into
/// the `ScalarBuffer`), so the logical offset alone cannot detect a slice. The
/// honest signal is at the buffer level: a non-sliced array's `ScalarBuffer`
/// covers its **entire** inner `Buffer` exactly — `ptr_offset() == 0` AND
/// `inner.len() == values.len() * size_of::<Native>()`. Any deviation means the
/// values view points into a larger parent buffer (a slice), which we reject so
/// we never upload aliased parent-buffer data (T-03-02). The reported `offset`
/// is the element offset into the parent buffer.
fn validate_no_offset<T>(arr: &PrimitiveArray<T>) -> Result<(), BridgeError>
where
    T: ArrowPrimitiveType,
{
    // Cheap logical-offset check first (covers wrappers that do surface it).
    if arr.offset() != 0 {
        return Err(BridgeError::Offset {
            offset: arr.offset(),
        });
    }
    let elem = size_of::<T::Native>();
    let values = arr.values();
    let inner = values.inner();
    let byte_offset = inner.ptr_offset();
    let covers_whole_buffer = byte_offset == 0 && inner.len() == values.len() * elem;
    if !covers_whole_buffer {
        // Report the element offset; for a from-the-start slice the byte_offset
        // is 0 but the parent buffer is longer, so offset is 0 yet we still
        // reject (the values view does not own the whole buffer). `elem` is the
        // size of a float type (4 or 8), never 0, so the division is total.
        let offset = byte_offset / elem;
        return Err(BridgeError::Offset { offset });
    }
    Ok(())
}

/// Reject any null entries before any transmute.
///
/// A nullable array's null slots hold meaningless values; uploading them would
/// produce silent wrong results (T-03-03). We require a fully-valid, null-free
/// buffer.
fn validate_no_nulls<A: Array>(arr: &A) -> Result<(), BridgeError> {
    let null_count = arr.null_count();
    if null_count != 0 || arr.nulls().is_some_and(|n| n.null_count() != 0) {
        return Err(BridgeError::HasNulls { null_count });
    }
    Ok(())
}

/// Alignment/size-checked reinterpretation of a raw byte buffer as `&[F]`.
///
/// This is the final gate before any data reaches device upload: it maps the
/// recoverable `bytemuck::try_cast_slice` error (A7 — never panics) to a typed
/// [`BridgeError::Misaligned`]. Exposed publicly so the misalignment class can
/// be tested directly against a deliberately misaligned `&[u8]` (a correctly
/// constructed Arrow array is always element-aligned and never trips this path
/// at runtime, but defense-in-depth keeps the check on the real ingress too).
pub fn cast_validated<F: Pod>(bytes: &[u8]) -> Result<&[F], BridgeError> {
    bytemuck::try_cast_slice::<u8, F>(bytes).map_err(|e| BridgeError::Misaligned {
        reason: e.to_string(),
    })
}

/// Upload an already-validated `&[F]` into a fresh CubeCL device buffer,
/// returning the device handle.
///
/// HONEST SEMANTICS (A3): this performs exactly **one** host copy
/// (`bytemuck::cast_slice(...).to_vec()` → `Bytes::from_bytes_vec`) because
/// cubecl 0.10 has no borrow/no-copy `Bytes` constructor. It is "validated
/// single-upload", not literal host zero-copy. Callers MUST pass a slice that
/// already came from [`validate_f32`] / [`validate_f64`]; this function does not
/// re-validate (the type system carries no offset/null information on a `&[F]`).
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn upload<F: Pod>(
    client: &crate::runtime::Client,
    validated: &[F],
) -> cubecl::server::Handle {
    // Single host copy into an owned byte Vec, then hand ownership to CubeCL.
    let byte_vec: Vec<u8> = bytemuck::cast_slice::<F, u8>(validated).to_vec();
    client.create(cubecl::bytes::Bytes::from_bytes_vec(byte_vec))
}
