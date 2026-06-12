//! Symmetric eigendecomposition host API (PRIM-05) — `A = V·diag(w)·Vᵀ` for a
//! square SYMMETRIC `A` (`n×n`), returning the device-resident eigenvalues `w`
//! (length `n`, DESCENDING — D-04) and eigenvectors `V` (`n×n`, eigenvectors as
//! columns). Drives the two-sided cyclic Jacobi sweep kernel
//! ([`mlrs_kernels::jacobi_eig_sweep`]) for the diagonalisation, then sorts the
//! converged diagonal descending on the host.
//!
//! ## The covariance/Gram feeder + buffer reuse (D-11 gate 2)
//! The eig path's only v1 feeder is the symmetric-by-construction covariance
//! Gram (`prims/covariance.rs`), so `A` is TRUSTED symmetric (D-06): this API
//! validates SQUARENESS but never forms `(A+Aᵀ)/2`. The optional `out` buffer
//! lets the caller thread the covariance/GEMM output handle straight through as
//! the kernel's working input — mirroring covariance's own "Gram reuses the GEMM
//! buffer" reuse — so the `full` PCA path does not allocate a parallel `n²`
//! matrix (D-11 gate 2, load-bearing for the Plan-05 memory gate). When `out` is
//! supplied it is copied into the kernel's `a_in` working buffer (the kernel
//! writes only `w`/`V`, leaving the caller's buffer the eig INPUT); when `None`
//! the input array is used directly.
//!
//! ## In-kernel convergence (D-11 gate 3)
//! The two-sided sweep loop — including the off-diagonal-norm convergence test —
//! runs entirely inside the single kernel launch (NO host round-trip between
//! sweeps). This API reads back ONLY the tiny length-`n` eigenvalue diagonal,
//! the `n×n` `V`, and the length-2 info array for the host-side descending sort
//! + the convergence check; it performs no read-back of intermediate sweeps.
//!
//! ## Descending sort (D-04) + convergence failure (D-12)
//! `np.linalg.eigh` returns eigenvalues ASCENDING; the device eig sorts them
//! DESCENDING so estimators inherit the right order. The host performs an `O(n)`
//! selection sort of the converged diagonal post-convergence (A4 — this is the
//! final sort, NOT the convergence loop the D-11 gate 3 concerns) and permutes
//! the eigenvector columns to match. If the kernel hit the sweep cap without
//! driving the off-diagonal norm below threshold, this API returns
//! [`PrimError::NotConverged`] rather than a silently-unconverged result.
//!
//! Tests live in `crates/mlrs-backend/tests/eig_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::jacobi_eig_sweep;
use mlrs_kernels::MAX_DIM;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Off-diagonal threshold scale factor `c` in `conv_thr = c · ε_F · ‖A‖_F ·
/// sqrt(pairs)` (D-12). `8` holds 1e-5 across the D-08 sweep while staying
/// reachable in f32 (mirrors the SVD host).
const THRESHOLD_SCALE: f64 = 8.0;

/// Max-sweep cap (D-12). Cyclic Jacobi converges quadratically (~10 sweeps for
/// the small symmetric covariance Gram); 30 is generous headroom (Pitfall 5).
const MAX_SWEEPS: u32 = 30;

/// Compute the symmetric eigendecomposition of `a` (`n × n`, row-major,
/// TRUSTED symmetric — D-06), returning the device-resident `(w, V)`: `w` the
/// length-`n` eigenvalues DESCENDING (D-04), `V` the `n × n` eigenvector matrix
/// (column-major, eigenvectors as columns).
///
/// - `a` is the row-major `n × n` symmetric matrix. Squareness is validated
///   (`a.len() == n*n`, and `n ≤ MAX_DIM`) BEFORE any unsafe launch (ASVS V5 /
///   T-03-04-01); a non-square geometry returns [`PrimError::NotSquare`].
/// - `out`, when supplied, is the covariance/GEMM output buffer reused as the
///   kernel's working input (D-11 gate 2): it must be the `n × n` operand. When
///   `None`, `a` is used directly.
/// - Non-convergence within the sweep cap returns [`PrimError::NotConverged`].
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log` (f64 runs on cpu,
/// skips on rocm — D-07).
#[allow(clippy::type_complexity)]
pub fn eig<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    n: usize,
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
    // --- ASVS V5 / T-03-04-01: validate squareness BEFORE any unsafe launch
    //     (D-06 trusts symmetry but validates the shape). ---
    validate_geometry(a.len(), n, out.as_ref().map(DeviceArray::len))?;

    let elem = size_of::<F>();

    // --- Working input buffer (D-11 gate 2 — covariance/GEMM buffer reuse).
    //     When the caller threads `out` through (the covariance Gram handle),
    //     that buffer is the kernel's `a_in` working input — no parallel n²
    //     allocation. The kernel only READS a_in (it writes w/V), so the
    //     caller's buffer is left intact as the eig input. When `out` is None
    //     we read directly from `a`. ---
    let (a_in_handle, a_in_owned) = match out {
        Some(buf) => (buf.handle().clone(), Some(buf)),
        None => (a.handle().clone(), None),
    };

    // --- Acquire the device-resident outputs: w (length n), V (n×n col-major),
    //     and the tiny info array [sweeps, residual]. ---
    let w_handle = pool.acquire(n * elem);
    let v_handle = pool.acquire(n * n * elem);
    let info_handle = pool.acquire(2 * elem);

    let client = pool.client().clone();
    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim {
        x: n as u32,
        y: 1,
        z: 1,
    };

    let (skip_thr, conv_thr) = compute_thresholds::<F>(pool, a, n * n, n);

    // SAFETY: lengths are the carried/validated element counts (n*n, n, n*n, 2),
    // NEVER raw caller geometry; the kernel bounds every loop by the runtime `n`
    // and idles units with `i >= n` (mitigates T-03-04-01 / T-03-04-03, the OOB
    // device-read threat, ASVS V5).
    let a_in_arg = unsafe { ArrayArg::from_raw_parts(a_in_handle, n * n) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_handle.clone(), n) };
    let v_arg = unsafe { ArrayArg::from_raw_parts(v_handle.clone(), n * n) };
    let info_arg = unsafe { ArrayArg::from_raw_parts(info_handle.clone(), 2) };

    jacobi_eig_sweep::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        a_in_arg,
        w_arg,
        v_arg,
        info_arg,
        n as u32,
        skip_thr,
        conv_thr,
        MAX_SWEEPS,
    );

    // The reused `out` working buffer (if any) is now consumed by the launch;
    // release it back to the pool (the kernel only read it). When `out` was
    // None we never owned `a`, so nothing to release here.
    if let Some(buf) = a_in_owned {
        buf.release_into(pool);
    }

    // --- Convergence check (D-12): read the tiny info array. info[0] = sweeps
    //     run; info[1] = final off-diagonal norm. A cap hit without convergence
    //     surfaces NotConverged. ---
    let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 2);
    let info = info_dev.to_host(pool);
    info_dev.release_into(pool);
    let sweeps_run = host_to_f64(info[0]) as u32;
    let residual = host_to_f64(info[1]);
    if sweeps_run >= MAX_SWEEPS && residual.is_finite() && residual > host_to_f64(conv_thr) {
        // Release the converged-output handles before surfacing the error.
        DeviceArray::<ActiveRuntime, F>::from_raw(w_handle, n).release_into(pool);
        DeviceArray::<ActiveRuntime, F>::from_raw(v_handle, n * n).release_into(pool);
        return Err(PrimError::NotConverged {
            operand: "eig",
            max_sweeps: MAX_SWEEPS,
            residual,
        });
    }

    // --- Host-side descending sort (D-04) + eigenvector-column permute. We read
    //     back the small w (length n) and V (n×n) — both device-resident
    //     producers; the convergence loop already ran in-kernel (D-11 gate 3).
    //     This O(n) sort is the FINAL ordering, not the convergence loop. ---
    let w_dev = DeviceArray::<ActiveRuntime, F>::from_raw(w_handle, n);
    let v_dev = DeviceArray::<ActiveRuntime, F>::from_raw(v_handle, n * n);
    let w_host = w_dev.to_host(pool);
    let v_host = v_dev.to_host(pool); // column-major V (v[c*n + r] = V[r, c]).
    w_dev.release_into(pool);
    v_dev.release_into(pool);

    let w64: Vec<f64> = w_host.iter().map(|&x| host_to_f64(x)).collect();

    // Descending order of eigenvalues with a permutation; permute V columns.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| w64[j].partial_cmp(&w64[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut w_sorted: Vec<F> = vec![F::from_int(0i64); n];
    let mut v_sorted: Vec<F> = vec![F::from_int(0i64); n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        w_sorted[new_j] = f64_to_host::<F>(w64[old_j]);
        // V is column-major (v_host[c*n + r] = V[r, c]); move column old_j to
        // new_j, preserving column-major layout.
        for r in 0..n {
            v_sorted[new_j * n + r] = v_host[old_j * n + r];
        }
    }

    let w_final = DeviceArray::from_host(pool, &w_sorted);
    let v_final = DeviceArray::from_host(pool, &v_sorted);
    Ok((w_final, v_final))
}

/// Validate the eig operand geometry (ASVS V5 / T-03-04-01). `a` must be a
/// square `n × n`: `a.len() == n*n`. The single-cube kernel stages the `n × n`
/// `A` + `V` in shared memory capped at `MAX_DIM`, so `n ≤ MAX_DIM` is required.
/// A non-square (or over-cap) geometry is rejected with [`PrimError::NotSquare`]
/// BEFORE any unsafe launch (D-06 trusts symmetry but validates squareness).
fn validate_geometry(a_len: usize, n: usize, out_len: Option<usize>) -> Result<(), PrimError> {
    // Squareness: a.len() must equal n*n. A mismatch means the caller's declared
    // order does not describe a square matrix.
    if n == 0 || n.checked_mul(n).map(|v| v != a_len).unwrap_or(true) {
        // Report the implied (rows, cols): a length that is not n*n is non-square.
        return Err(PrimError::NotSquare {
            operand: "eig",
            rows: n,
            cols: if n == 0 { 0 } else { a_len / n.max(1) },
        });
    }
    if n > MAX_DIM as usize {
        // Geometry the single-cube kernel cannot stage; reject rather than
        // overflow shared memory at launch.
        return Err(PrimError::NotSquare {
            operand: "eig",
            rows: n,
            cols: n,
        });
    }
    // The reused `out` buffer (D-11 gate 2) must itself be the n×n operand.
    if let Some(o) = out_len {
        if o != n * n {
            return Err(PrimError::NotSquare {
                operand: "eig.out",
                rows: n,
                cols: if n == 0 { 0 } else { o / n.max(1) },
            });
        }
    }
    Ok(())
}

/// Compute the `(skip_thr, conv_thr)` pair (D-12), mirroring the SVD host.
/// `‖A‖_F` is the input's Frobenius norm; `ε_F` the per-dtype machine epsilon;
/// `pairs = n(n-1)/2`.
///   - `skip_thr = ε_F · ‖A‖_F` — TINY, so rotations are essentially never
///     skipped (a loose skip bound stalls convergence — Pitfall 5).
///   - `conv_thr = 8 · ε_F · ‖A‖_F · sqrt(pairs)` — the convergence-break bound,
///     scaled by `sqrt(pairs)` to clear the ACCUMULATED f32 rounding floor.
/// Reads the input back ONCE to form `‖A‖_F` on the host — a pre-launch scale
/// estimate, NOT a mid-sweep round-trip (the convergence loop stays in-kernel).
fn compute_thresholds<F>(
    pool: &BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    len: usize,
    n: usize,
) -> (F, F)
where
    F: Float + CubeElement + Pod,
{
    let host = a.to_host(pool);
    let mut sumsq = 0.0f64;
    for i in 0..len {
        let v = host_to_f64(host[i]);
        sumsq += v * v;
    }
    let fro = sumsq.sqrt();
    let eps = match size_of::<F>() {
        4 => f32::EPSILON as f64,
        _ => f64::EPSILON,
    };
    let pairs = (n * n.saturating_sub(1)) as f64 / 2.0;
    let skip_thr = (eps * fro).max(eps);
    let conv_thr = (THRESHOLD_SCALE * eps * fro * pairs.max(1.0).sqrt()).max(skip_thr);
    (f64_to_host::<F>(skip_thr), f64_to_host::<F>(conv_thr))
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine / finalize.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("eig is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("eig is f32/f64 only"),
    }
}
