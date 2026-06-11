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
//! ## Transpose without a transpose buffer (D-06)
//! `transa` / `transb` are applied by swapping the operand's logical
//! `(shape, strides)` via `InputBinding::swap_dims` — the device reads the
//! transposed view directly, no materialized transpose buffer.
//!
//! ## f64 precision (Pitfall 2 — recorded)
//! `cubek_matmul`'s default `MatmulPrecision for f64` uses an f32 stage/register
//! profile, but we build [`MatmulElems`] via `from_globals`, which keeps the
//! accumulator at the f64 global dtype for non-f16/bf16 outputs — so f64 GEMM
//! accumulates in f64. The f64 path is capability-gated via `skip_f64_with_log`.
//!
//! Tests live in `crates/mlrs-backend/tests/gemm_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubePrimitive, Numeric, TensorBinding};
use cubecl::std::tensor::TensorHandle;
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_matmul::launch::{Strategy, launch_ref};
use cubek_std::InputBinding;

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

    let out_len = m * n;
    let elem_size = size_of::<F>();
    let dtype = F::as_type_native_unchecked().storage_type();

    // --- Build cubek-matmul bindings from the device handles (D-06 transpose
    // is a logical (shape, strides) swap, no transpose buffer). ---
    // lhs logical shape is (m, k); under transa the STORED buffer is (k, m) and
    // we present it swapped so the device reads the transposed view.
    let a_stored_shape = if transa { vec![k, m] } else { vec![m, k] };
    let lhs_tensor =
        TensorHandle::<ActiveRuntime>::new_contiguous(a_stored_shape, a.handle().clone(), dtype);
    let mut lhs = InputBinding::new(lhs_tensor.binding(), dtype);
    if transa {
        lhs.swap_dims(0, 1);
    }

    // rhs logical shape is (k, n); under transb the STORED buffer is (n, k).
    let b_stored_shape = if transb { vec![n, k] } else { vec![k, n] };
    let rhs_tensor =
        TensorHandle::<ActiveRuntime>::new_contiguous(b_stored_shape, b.handle().clone(), dtype);
    let mut rhs = InputBinding::new(rhs_tensor.binding(), dtype);
    if transb {
        rhs.swap_dims(0, 1);
    }

    // Output buffer: reuse the caller's (D-11), else acquire m*n elems' bytes.
    let out_bytes = out_len * elem_size;
    let out_handle = match &out {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_bytes),
    };
    let out_tensor =
        TensorHandle::<ActiveRuntime>::new_contiguous(vec![m, n], out_handle.clone(), dtype);
    let out_binding: TensorBinding<ActiveRuntime> = out_tensor.binding();

    let global_dtypes = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    };
    // from_globals keeps the accumulator at the f64 global dtype for
    // non-f16/bf16 outputs (so f64 accumulates in f64 — Pitfall 2 mitigation).
    let mut dtypes = MatmulElems::from_globals(&global_dtypes);

    let client = pool.client().clone();
    launch_ref::<ActiveRuntime>(
        &Strategy::default(),
        &client,
        lhs,
        rhs,
        out_binding,
        &mut dtypes,
    )
    .expect("cubek-matmul launch for GEMM");

    // The result stays device-resident (D-05); the caller reads it back via
    // to_host / to_host_metered when needed.
    Ok(DeviceArray::from_raw(out_handle, out_len))
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
