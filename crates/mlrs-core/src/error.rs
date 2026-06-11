//! Typed error enums shared across the workspace (D-07 / D-10).
//!
//! [`BridgeError`] is the Arrow→CubeCL bridge's error type. It is defined here
//! in `mlrs-core` (the dependency-free foundation crate) so that Plan 03's
//! bridge in `mlrs-backend` can `return Err(BridgeError::...)` without a
//! reverse dependency. One variant per Arrow-violation class (D-07).

use thiserror::Error;

/// Errors raised when validating an Apache Arrow buffer before uploading it
/// into a CubeCL device buffer (Plan 03 bridge, FOUND-06 / D-07).
///
/// Each variant corresponds to a distinct Arrow invariant the bridge must
/// reject *before* any `unsafe` transmute, so a non-conforming array becomes a
/// recoverable typed error rather than undefined behaviour.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// The Arrow array has a non-zero logical offset (e.g. a sliced array),
    /// so `values()` does not start at the data the caller expects.
    #[error(
        "arrow array has non-zero offset {offset}; expected a non-sliced (offset == 0) array"
    )]
    Offset {
        /// The offending logical offset, in elements.
        offset: usize,
    },

    /// The Arrow array contains null entries; the bridge requires a fully
    /// valid (null-free) buffer.
    #[error("arrow array contains {null_count} null(s); a null-free buffer is required")]
    HasNulls {
        /// Number of null entries detected.
        null_count: usize,
    },

    /// The underlying byte buffer is not correctly aligned (or not size-
    /// divisible) for the target element type — surfaced from
    /// `bytemuck::try_cast_slice` rather than panicking (A7 / D-06).
    #[error(
        "arrow buffer is misaligned or wrongly sized for the target element type: {reason}"
    )]
    Misaligned {
        /// Human-readable detail from the failed cast.
        reason: String,
    },

    /// The Arrow data type did not match the element type the bridge expected
    /// (e.g. a `Float32Array` where `f64` was requested).
    #[error("arrow data type mismatch: expected {expected}, found {found}")]
    DataTypeMismatch {
        /// The element type the bridge requested.
        expected: String,
        /// The element type actually present.
        found: String,
    },
}
