//! GEMM host API (PRIM-01) — `C = op(A) · op(B)` with BLAS-style transpose
//! flags, device-resident in/out, pool-routed output buffer.
//!
//! ## Substrate: cubek-matmul wrap (Plan 02-01 Task-1 checkpoint RESOLVED)
//! The device matmul is delegated to `cubek-matmul` 0.2.0 — the
//! cubecl-0.10-compatible matmul source. The abandoned `cubecl-matmul`
//! (0.9.0-pre.5) and `cubecl-linalg` (0.5.0) pin incompatible cubecl-core lines
//! and cannot link against the workspace's `cubecl 0.10.0`. Host orchestration
//! (shape validation, pool-routed out buffer, the launch wrap) lives here in
//! `mlrs-backend`; `mlrs-kernels` stays feature-free (D-13), so NO
//! `gemm_kernel` is hand-written there.
//!
//! The full body (cubek-matmul binding construction + launch) lands in Task 5;
//! this module currently exposes the validated host signature so the Task-4
//! test scaffold compiles against it.
//!
//! Tests live in `crates/mlrs-backend/tests/gemm_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubePrimitive, Numeric};

use mlrs_core::PrimError;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Compute `C = op_a(A) · op_b(B)` on the device, returning a device-resident
/// [`DeviceArray`].
///
/// - `a` is the row-major `m × k` left operand (logical, pre-transpose); `b` is
///   the row-major `k × n` right operand. `transa` / `transb` apply a zero-copy
///   logical transpose to that operand (D-06).
/// - Shapes are validated against the operand element counts **before** any
///   launch (`m*k == a.len()`, `k*n == b.len()` — D-04 / T-0201-02); a mismatch
///   returns [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`].
/// - The `m × n` output is acquired from `pool` when `out` is `None`, else the
///   supplied buffer is reused (D-11). The result stays on the device — NO host
///   round-trip inside the API (D-05).
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn gemm<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (m, k): (usize, usize),
    b: &DeviceArray<ActiveRuntime, F>,
    (k2, n): (usize, usize),
    transa: bool,
    transb: bool,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Numeric + CubePrimitive + Pod,
{
    // --- D-04 / T-0201-02: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(a.len(), (m, k), b.len(), (k2, n), out.as_ref().map(DeviceArray::len))?;

    // Body (cubek-matmul binding construction + launch) lands in Task 5.
    let _ = (pool, transa, transb, out);
    todo!("GEMM device launch lands in Plan 02-01 Task 5 (cubek-matmul wrap)")
}

/// Validate GEMM operand geometry (D-04). Extracted so the Task-4 scaffold can
/// exercise the shape-rejection contract before the device body exists.
pub(crate) fn validate_geometry(
    a_len: usize,
    (m, k): (usize, usize),
    b_len: usize,
    (k2, n): (usize, usize),
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    if m.checked_mul(k).map(|v| v != a_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "lhs",
            rows: m,
            cols: k,
            len: a_len,
        });
    }
    if k2.checked_mul(n).map(|v| v != b_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "rhs",
            rows: k2,
            cols: n,
            len: b_len,
        });
    }
    if k != k2 {
        return Err(PrimError::DimMismatch {
            dim: "k",
            lhs: k,
            rhs: k2,
        });
    }
    if let Some(o) = out_len {
        let expect = m * n;
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out",
                rows: m,
                cols: n,
                len: o,
            });
        }
    }
    Ok(())
}
