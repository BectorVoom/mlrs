//! Cholesky/SPD-solve host API (D-02) — factor a square SPD `A` (`n × n`,
//! row-major, TRUSTED symmetric) as `A = L·Lᵀ` and solve `A·x = b` for `rhs`
//! right-hand-side columns, returning the device-resident solution `x`. Drives
//! the single-cube, in-kernel Cholesky+triangular-solve kernel
//! ([`mlrs_kernels::cholesky_solve`]); the entire factor→forward→back pipeline
//! runs in ONE launch with NO host round-trip between phases (D-11 gate 3).
//!
//! This is the single genuinely-new device primitive of Phase 4: Ridge's
//! normal-equations solve `(XᵀX + αI)·coef = Xᵀy` needs an SPD solve with no
//! Phase-2/3 analogue (D-02). It mirrors the [`crate::prims::eig`] host wrapper:
//! validate geometry BEFORE the unsafe launch (ASVS V5), thread an optional
//! reused working buffer through (D-11 gate 2 — Ridge passes the Gram `XᵀX`
//! buffer so no parallel `n²` allocation), launch the single cube, read back the
//! tiny `info` array, and surface a typed error on a non-SPD pivot.
//!
//! ## Validate-before-launch (ASVS V5 / T-04-02-01)
//! `a.len() == n*n` (else [`PrimError::NotSquare`]), `n ≤ MAX_DIM` (the
//! single-cube kernel stages `L` in shared memory capped at `MAX_DIM`; else
//! `NotSquare`), and `b.len() == n*rhs` (else [`PrimError::ShapeMismatch`]) are
//! all checked BEFORE any `unsafe { ArrayArg::from_raw_parts }`, so a wrong shape
//! is a recoverable typed error rather than an out-of-bounds device read.
//!
//! ## Non-SPD guard (T-04-02-02 / RESEARCH Pitfall 4)
//! A near-singular or indefinite `A` produces a non-positive diagonal pivot under
//! the Cholesky square root. The kernel flags it in `info_out` (NEVER emits NaN);
//! this host reads `info` back and returns [`PrimError::NotPositiveDefinite`]
//! with the offending pivot index/value rather than a NaN-poisoned solution.
//!
//! Tests live in `crates/mlrs-backend/tests/cholesky_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::cholesky_solve as cholesky_solve_kernel;
use mlrs_kernels::MAX_DIM;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Factor a square SPD `a` (`n × n`, row-major, TRUSTED symmetric — D-06) and
/// solve `A·x = b` for `rhs` right-hand-side columns, returning the device-
/// resident solution `x` (`n × rhs`, row-major).
///
/// - `a` is the row-major `n × n` SPD matrix. Squareness (`a.len() == n*n`,
///   `n ≤ MAX_DIM`) is validated BEFORE any unsafe launch (ASVS V5 /
///   T-04-02-01); a non-square geometry returns [`PrimError::NotSquare`].
/// - `b` is the row-major `n × rhs` right-hand side; `b.len() == n*rhs` is
///   validated, else [`PrimError::ShapeMismatch`].
/// - `out`, when supplied, is the covariance/GEMM `n × n` working buffer reused
///   as the kernel's `a_in` input (D-11 gate 2 — Ridge threads the Gram buffer so
///   no parallel `n²` allocation). When `None`, `a` is used directly.
/// - A non-SPD diagonal pivot returns [`PrimError::NotPositiveDefinite`].
///
/// Generic over `F` (`f32` / `f64`); the f64 path is capability-gated by the
/// caller via `skip_f64_with_log` (f64 runs on cpu, skips on rocm — D-07).
pub fn cholesky_solve<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    b: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    rhs: usize,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (x, l) = cholesky_solve_with_factor(pool, a, b, n, rhs, out)?;
    // The L factor is internal scratch for this public path; release it.
    l.release_into(pool);
    Ok(x)
}

/// Like [`cholesky_solve`] but ALSO returns the device-resident lower Cholesky
/// factor `L` (`n × n`, row-major, strictly-upper entries 0) written by the
/// kernel. This is the unambiguous source of `L` for the standalone
/// `‖L·Lᵀ − A‖` reconstruction invariant — the test reads back the
/// KERNEL-EMITTED factor rather than re-deriving it on the host (resolving the
/// prior "Claude's Discretion" L-source ambiguity).
///
/// Returns `(x, L)`. The caller owns both device arrays and is responsible for
/// releasing them back into the pool.
#[allow(clippy::type_complexity)]
pub fn cholesky_solve_with_factor<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    b: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    rhs: usize,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-04-02-01: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(a.len(), b.len(), n, rhs, out.as_ref().map(DeviceArray::len))?;

    let elem = size_of::<F>();

    // --- Working input buffer (D-11 gate 2 — Gram/GEMM buffer reuse). When the
    //     caller threads `out` through (the Gram XᵀX handle), that buffer is the
    //     kernel's `a_in` working input — no parallel n² allocation. The kernel
    //     only READS a_in (it writes x/L/info), so the caller's buffer is left
    //     intact as the solve input. When `out` is None we read directly from
    //     `a`. ---
    let (a_in_handle, a_in_owned) = match out {
        Some(buf) => (buf.handle().clone(), Some(buf)),
        None => (a.handle().clone(), None),
    };

    // --- Acquire the device-resident outputs: x (n×rhs), the lower factor L
    //     (n×n), and the tiny length-3 info array
    //     [non_spd_flag, pivot_index, pivot_value]. ---
    let x_handle = pool.acquire(n * rhs * elem);
    let l_handle = pool.acquire(n * n * elem);
    let info_handle = pool.acquire(3 * elem);

    let client = pool.client().clone();
    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim {
        x: n as u32,
        y: 1,
        z: 1,
    };

    // SAFETY: lengths are the carried/validated element counts (n*n, n*rhs,
    // n*rhs, n*n, 2), NEVER raw caller geometry; the kernel bounds every loop by
    // the runtime `n`/`rhs` and only unit 0 acts (mitigates T-04-02-01 /
    // T-04-02-03, the OOB device-read threat, ASVS V5).
    let a_in_arg = unsafe { ArrayArg::from_raw_parts(a_in_handle, n * n) };
    let b_arg = unsafe { ArrayArg::from_raw_parts(b.handle().clone(), n * rhs) };
    let x_arg = unsafe { ArrayArg::from_raw_parts(x_handle.clone(), n * rhs) };
    let l_arg = unsafe { ArrayArg::from_raw_parts(l_handle.clone(), n * n) };
    let info_arg = unsafe { ArrayArg::from_raw_parts(info_handle.clone(), 3) };

    cholesky_solve_kernel::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        a_in_arg,
        b_arg,
        x_arg,
        l_arg,
        info_arg,
        n as u32,
        rhs as u32,
    );

    // The reused `out` working buffer (if any) is now consumed by the launch;
    // release it back to the pool (the kernel only read it). When `out` was None
    // we never owned `a`, so nothing to release here.
    if let Some(buf) = a_in_owned {
        buf.release_into(pool);
    }

    // --- Non-SPD guard (T-04-02-02 / D-12): read the tiny info array.
    //     info[0] < 0 ⇒ a non-positive pivot was hit; info[1] is the failing
    //     diagonal index and info[2] is the actual non-positive pivot value.
    //     Surface NotPositiveDefinite rather than a NaN result. ---
    let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 3);
    let info = info_dev.to_host(pool);
    info_dev.release_into(pool);
    let flag = host_to_f64(info[0]);
    if flag < 0.0 {
        // Release the (unused) output handles before surfacing the error.
        DeviceArray::<ActiveRuntime, F>::from_raw(x_handle, n * rhs).release_into(pool);
        DeviceArray::<ActiveRuntime, F>::from_raw(l_handle, n * n).release_into(pool);
        let pivot_index = host_to_f64(info[1]) as usize;
        let pivot_value = host_to_f64(info[2]);
        return Err(PrimError::NotPositiveDefinite {
            operand: "cholesky",
            pivot_index,
            pivot_value,
        });
    }

    let x = DeviceArray::<ActiveRuntime, F>::from_raw(x_handle, n * rhs);
    let l = DeviceArray::<ActiveRuntime, F>::from_raw(l_handle, n * n);
    Ok((x, l))
}

/// Validate the Cholesky operand geometry (ASVS V5 / T-04-02-01). `a` must be a
/// square `n × n` (`a.len() == n*n`); the single-cube kernel stages `L` in shared
/// memory capped at `MAX_DIM`, so `n ≤ MAX_DIM` is required (both rejected with
/// [`PrimError::NotSquare`]). `b` must be `n × rhs` (`b.len() == n*rhs`, else
/// [`PrimError::ShapeMismatch`]). The optional reused `out` buffer must itself be
/// the `n × n` operand. All checks run BEFORE any unsafe launch.
fn validate_geometry(
    a_len: usize,
    b_len: usize,
    n: usize,
    rhs: usize,
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    // Squareness: a.len() must equal n*n.
    if n == 0 || n.checked_mul(n).map(|v| v != a_len).unwrap_or(true) {
        return Err(PrimError::NotSquare {
            operand: "cholesky",
            rows: n,
            cols: if n == 0 { 0 } else { a_len / n.max(1) },
        });
    }
    // Over-cap: the single-cube kernel cannot stage L > MAX_DIM in shared memory.
    if n > MAX_DIM as usize {
        return Err(PrimError::NotSquare {
            operand: "cholesky",
            rows: n,
            cols: n,
        });
    }
    // RHS geometry: b.len() must equal n*rhs.
    if rhs == 0 || n.checked_mul(rhs).map(|v| v != b_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "cholesky.b",
            rows: n,
            cols: rhs,
            len: b_len,
        });
    }
    // The reused `out` working buffer (D-11 gate 2) must itself be the n×n operand.
    if let Some(o) = out_len {
        if o != n * n {
            return Err(PrimError::NotSquare {
                operand: "cholesky.out",
                rows: n,
                cols: if n == 0 { 0 } else { o / n.max(1) },
            });
        }
    }
    Ok(())
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side info decoding.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cholesky is f32/f64 only"),
    }
}
