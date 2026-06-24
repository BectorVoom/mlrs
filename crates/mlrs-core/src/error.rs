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

/// Errors raised when validating the geometry of a compute primitive's operands
/// before launching a device kernel (Phase 2 primitives, D-04 / ASVS V5).
///
/// Every variant corresponds to a caller-supplied-geometry violation the
/// primitive must reject *before* any `unsafe` kernel launch, so a wrong shape
/// becomes a recoverable typed error rather than an out-of-bounds device read.
/// One variant per violation class (D-07), mirroring [`BridgeError`].
#[derive(Debug, Error)]
pub enum PrimError {
    /// A declared `(rows, cols)` geometry does not match the operand's element
    /// count: `rows * cols != len`. Carries the offending values plus a label
    /// naming the operand (e.g. `"lhs"`, `"rhs"`, `"out"`) for diagnosis.
    #[error(
        "primitive '{operand}' shape mismatch: rows({rows}) * cols({cols}) = {} != len({len})",
        rows * cols
    )]
    ShapeMismatch {
        /// Which operand failed validation (e.g. `"lhs"` / `"rhs"` / `"out"`).
        operand: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
        /// Actual element count of the operand buffer.
        len: usize,
    },

    /// Two operands disagree on a shared dimension that must match for the
    /// operation to be defined — e.g. a GEMM where `lhs` is `m×k` but `rhs` is
    /// `k'×n` with `k != k'` (the contraction dimension is incompatible).
    #[error(
        "primitive dimension mismatch ({dim}): lhs declares {lhs}, rhs declares {rhs}"
    )]
    DimMismatch {
        /// Name of the disagreeing dimension (e.g. `"k"` for the GEMM
        /// contraction dimension).
        dim: &'static str,
        /// The dimension value the lhs operand declared.
        lhs: usize,
        /// The dimension value the rhs operand declared.
        rhs: usize,
    },

    /// A primitive that requires a square matrix (e.g. the symmetric
    /// eigendecomposition, which trusts symmetry but validates squareness —
    /// D-06 / ASVS V5) was handed a `rows != cols` geometry. Rejected *before*
    /// any `unsafe` kernel launch so a wrong shape is a recoverable typed error,
    /// not an out-of-bounds device read. Carries a label naming the operand.
    #[error(
        "primitive '{operand}' must be square: rows({rows}) != cols({cols})"
    )]
    NotSquare {
        /// Which operand failed the squareness check (e.g. `"input"`).
        operand: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },

    /// An iterative primitive (the Jacobi SVD/eig sweep) failed to drive the
    /// off-diagonal norm below its internal threshold within the max-sweep cap
    /// (D-12 — convergence constants are internal, not public API). Carries the
    /// operand label, the sweep cap that was hit, and the final off-diagonal
    /// norm so the caller can diagnose a pathological input.
    #[error(
        "primitive '{operand}' did not converge within {max_sweeps} sweeps (off-diagonal norm {residual:e})"
    )]
    NotConverged {
        /// Which primitive failed to converge (e.g. `"svd"` / `"eig"`).
        operand: &'static str,
        /// The max-sweep cap that was reached without converging (D-12).
        max_sweeps: u32,
        /// The final off-diagonal Frobenius norm at the sweep cap.
        residual: f64,
    },

    /// A primitive that factorizes a symmetric POSITIVE-DEFINITE matrix (the
    /// Phase-4 Cholesky normal-equations solve that Ridge consumes — D-02)
    /// encountered a non-positive pivot during factorization: the matrix is not
    /// actually SPD, so its Cholesky factor does not exist. The kernel flags the
    /// offending pivot (via its `info` array) and the host surfaces it here
    /// rather than returning a NaN-poisoned factor (RESEARCH Pitfall 4). Carries
    /// the operand label, the diagonal index where the pivot went non-positive,
    /// and the pivot value (the would-be `√` argument) for diagnosis.
    #[error(
        "primitive '{operand}' is not positive-definite: non-positive pivot {pivot_value:e} \
         at diagonal index {pivot_index} (matrix is not SPD; Cholesky factor does not exist)"
    )]
    NotPositiveDefinite {
        /// Which primitive failed the SPD check (e.g. `"cholesky"`).
        operand: &'static str,
        /// The diagonal index where the running pivot became non-positive.
        pivot_index: usize,
        /// The non-positive pivot value (the negative/zero `√` argument).
        pivot_value: f64,
    },

    /// A `usize` multiplication that sizes a device buffer (e.g. `rows * cols`
    /// for an `n×n` block) overflowed, so the requested geometry cannot be
    /// allocated. Rejected *before* any `unsafe` kernel launch (the `checked_mul`
    /// guard) so an out-of-range geometry is a recoverable typed error rather than
    /// a fabricated `usize::MAX` length in a `ShapeMismatch`. Carries the operand
    /// label and the two operands whose product overflowed.
    #[error(
        "primitive '{operand}' geometry overflows usize: {lhs} * {rhs} does not fit in usize"
    )]
    Overflow {
        /// Which operand's geometry overflowed (e.g. `"d"`, `"cosine_distance_matrix"`).
        operand: &'static str,
        /// The left operand of the overflowing multiplication.
        lhs: usize,
        /// The right operand of the overflowing multiplication.
        rhs: usize,
    },
}
