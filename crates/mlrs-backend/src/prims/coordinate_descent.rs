//! `prims::coordinate_descent` — host-driven coordinate-descent loop for the
//! Lasso / ElasticNet solver (LINEAR-03/04, D-03 shared CD path).
//!
//! Implements RESEARCH §"Pattern 2" (host-driven iteration, D-10): the HOST owns
//! the cyclic coordinate loop and the soft-threshold scalar math; the DEVICE
//! carries the n-heavy work (`col_dot` for `t = X_j·R`, `residual_axpy` for the
//! residual update, `enet_gap` for the duality gap). The solver buffers
//! (`x`, `r`, `y` on device; `w` host-side scalar state) are acquired ONCE before
//! the loop and reused every iteration; EXACTLY ONE scalar (the duality gap) is
//! read back per outer convergence check (the D-10 bounded-allocation exception —
//! `prims::cholesky`'s validate→launch→scalar-readback shape).
//!
//! ## Penalty + stopping constants (pinned, Pitfall 1/2)
//! The caller passes the ALREADY-un-normalized penalties
//! `l1_reg = α·l1_ratio·n_samples` and `l2_reg = α·(1−l1_ratio)·n_samples`
//! (`_coordinate_descent.py:781-782`); the soft-threshold is
//! `w_j = sign(t)·max(|t| − l1_reg, 0)/(‖X_j‖² + l2_reg)` and the stop is
//! `gap ≤ tol·‖y‖²` (`tol *= dot(y,y)` ONCE before the loop — `_cd_fast.pyx`).
//! `max_iter` defaults to 1000. NO Gap-Safe screening is reproduced (Anti-Pattern;
//! plain cyclic CD reaches the identical `coef_`).
//!
//! ## Validate-before-launch (ASVS V5 / T-05-05-01)
//! `n*d == x.len()` and `y.len() == n` are checked → [`PrimError::ShapeMismatch`]
//! BEFORE any `unsafe` launch, so a wrong geometry is a recoverable typed error
//! rather than an out-of-bounds device read. A zero column
//! (`norm2_cols[j] == 0`) is SKIPPED (T-05-05-02 — no divide-by-zero / NaN).
//!
//! Tests live in `crates/mlrs-backend/tests/cd_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::coordinate::{cd_col_dot, cd_enet_gap, cd_residual_axpy};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Default sklearn coordinate-descent iteration cap (`max_iter`).
pub const CD_MAX_ITER: usize = 1000;

/// Solve the ElasticNet/Lasso coordinate-descent problem on the CENTERED design
/// `x` (`n × d`, row-major) and target `y` (length `n`), returning the
/// device-resident coefficient vector `coef` (length `d`).
///
/// Reproduces sklearn's `enet_coordinate_descent` (`_cd_fast.pyx`, de-screened):
/// cyclic soft-threshold updates with the un-normalized penalties `l1_reg`
/// (= `α·l1_ratio·n`) and `l2_reg` (= `α·(1−l1_ratio)·n`), stopping on the
/// duality gap `≤ tol·‖y‖²` or after `max_iter` passes. Lasso is the `l2_reg = 0`
/// case (`l1_ratio = 1`); ElasticNet uses both (D-03 shared path).
///
/// - Geometry (`n*d == x.len()`, `y.len() == n`) is validated BEFORE any unsafe
///   launch (ASVS V5 / T-05-05-01); a wrong shape returns
///   [`PrimError::ShapeMismatch`].
/// - `r` (residual, = `y` at `w = 0`) and `x`/`y` device buffers are acquired
///   ONCE and reused every iteration (D-10 bounded allocation); `w` is host-side
///   scalar state (the soft-threshold is one scalar per coordinate).
/// - EXACTLY ONE scalar (the duality gap) is read back per outer convergence
///   check (D-10; T-05-05-03).
/// - A zero column (`‖X_j‖² == 0`) is skipped to avoid divide-by-zero
///   (T-05-05-02).
///
/// Generic over `F` (`f32` / `f64`); the f64 path is capability-gated by the
/// caller via `skip_f64_with_log` (f64 runs on cpu, skips on rocm — D-07).
#[allow(clippy::too_many_arguments)]
pub fn cd_solve<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    l1_reg: f64,
    l2_reg: f64,
    tol: f64,
    max_iter: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-05-05-01: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(x.len(), y.len(), n, d)?;

    let elem = size_of::<F>();
    let client = pool.client().clone();

    // --- ‖X_j‖² (column SumSq), computed ONCE on the host from a single X
    //     read-back (this is a pre-loop setup read, NOT a per-iteration array
    //     readback — D-10). norm2_cols + the host w/y vectors are the reused
    //     solver state. ---
    let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let y_host: Vec<f64> = y.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let mut norm2_cols = vec![0.0f64; d];
    for j in 0..d {
        let mut s = 0.0f64;
        for i in 0..n {
            let v = x_host[i * d + j];
            s += v * v;
        }
        norm2_cols[j] = s;
    }

    // tol_scaled = tol * dot(y, y) ONCE before the loop (Pitfall 2).
    let y_norm2: f64 = y_host.iter().map(|&v| v * v).sum();
    let tol_scaled = tol * y_norm2;

    // --- Solver buffers acquired ONCE, reused every iteration (D-10). `r` starts
    //     at `y` (residual at `w = 0`); `w` is host-side scalar coefficient
    //     state. ---
    let r_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y.to_host(pool));
    let gap_handle = pool.acquire(elem); // length-1 scalar gap output, reused.
    let dot_handle = pool.acquire(elem); // length-1 scalar col-dot output, reused.

    let mut w = vec![0.0f64; d];

    let cube1 = CubeCount::Static(1, 1, 1);
    let dim1 = CubeDim { x: 1, y: 1, z: 1 };
    // residual_axpy is the per-row map (one unit per row, ceil-div over 256).
    let axpy_block = 256u32;
    let axpy_cubes = ((n as u32) + axpy_block - 1) / axpy_block.max(1);
    let axpy_count = CubeCount::Static(axpy_cubes.max(1), 1, 1);
    let axpy_dim = CubeDim {
        x: axpy_block,
        y: 1,
        z: 1,
    };

    let max_iter = if max_iter == 0 { CD_MAX_ITER } else { max_iter };

    for n_iter in 0..max_iter {
        let mut w_max = 0.0f64;
        let mut d_w_max = 0.0f64;

        for j in 0..d {
            // T-05-05-02: skip a zero column (divide-by-zero / NaN guard).
            if norm2_cols[j] == 0.0 {
                continue;
            }

            // t = X[:,j]·R + w_j_old·‖X_j‖²  (device col-dot + host scalar fuse).
            let dot_out = unsafe { ArrayArg::from_raw_parts(dot_handle.clone(), 1) };
            let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) };
            let r_arg = unsafe { ArrayArg::from_raw_parts(r_dev.handle().clone(), n) };
            cd_col_dot::launch::<F, ActiveRuntime>(
                &client,
                cube1.clone(),
                dim1,
                x_arg,
                r_arg,
                dot_out,
                n as u32,
                d as u32,
                j as u32,
            );
            // `from_raw` holds a CLONE of the loop-invariant `dot_handle`; the
            // wrapper drops at end of scope (just one handle clone) while
            // `dot_handle` stays owned + reused next coordinate (no release —
            // it is loop-invariant scratch, freed once after the loop).
            let dot_dev = DeviceArray::<ActiveRuntime, F>::from_raw(dot_handle.clone(), 1);
            let xj_dot_r = host_to_f64(dot_dev.to_host(pool)[0]);

            let w_old = w[j];
            let t = xj_dot_r + w_old * norm2_cols[j];

            // Soft-threshold (host scalar): w_j = sign(t)·max(|t|−l1_reg,0)/(‖X_j‖²+l2_reg).
            let w_new = soft_threshold(t, l1_reg) / (norm2_cols[j] + l2_reg);
            w[j] = w_new;

            // If w_j changed: R += (w_j_old − w_j_new)·X[:,j]  (device residual axpy).
            let factor = w_old - w_new;
            if factor != 0.0 {
                let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) };
                let r_arg = unsafe { ArrayArg::from_raw_parts(r_dev.handle().clone(), n) };
                cd_residual_axpy::launch::<F, ActiveRuntime>(
                    &client,
                    axpy_count.clone(),
                    axpy_dim,
                    x_arg,
                    r_arg,
                    from_f64::<F>(factor),
                    n as u32,
                    d as u32,
                    j as u32,
                );
            }

            // sklearn's cheap host convergence-gate accumulators.
            let aw = w_new.abs();
            if aw > w_max {
                w_max = aw;
            }
            let dw = factor.abs();
            if dw > d_w_max {
                d_w_max = dw;
            }
        }

        // sklearn cheap host gate, then the ONE-scalar gap readback (D-10).
        let last_iter = n_iter + 1 == max_iter;
        let host_gate = w_max == 0.0 || (d_w_max / w_max) <= tol || last_iter;
        if host_gate {
            // Upload the current host `w` for the device gap kernel (length-d,
            // reused-shape; the gap kernel reads X/R/y/w device-side and emits
            // ONE scalar — no array readback).
            let w_f: Vec<F> = w.iter().map(|&v| from_f64::<F>(v)).collect();
            let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_f);

            let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) };
            let r_arg = unsafe { ArrayArg::from_raw_parts(r_dev.handle().clone(), n) };
            let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), n) };
            let w_arg = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
            let gap_arg = unsafe { ArrayArg::from_raw_parts(gap_handle.clone(), 1) };
            cd_enet_gap::launch::<F, ActiveRuntime>(
                &client,
                cube1.clone(),
                dim1,
                x_arg,
                r_arg,
                y_arg,
                w_arg,
                gap_arg,
                n as u32,
                d as u32,
                from_f64::<F>(l1_reg),
                from_f64::<F>(l2_reg),
            );
            let gap_dev = DeviceArray::<ActiveRuntime, F>::from_raw(gap_handle.clone(), 1);
            let gap = host_to_f64(gap_dev.to_host_metered(pool)[0]); // ← the ONE scalar (D-10).
            w_dev.release_into(pool);

            if gap <= tol_scaled {
                break;
            }
        }
    }

    // Release the reused loop scratch; materialize the host coef as the
    // device-resident result.
    r_dev.release_into(pool);
    pool.release(gap_handle, elem);
    pool.release(dot_handle, elem);

    let coef_f: Vec<F> = w.iter().map(|&v| from_f64::<F>(v)).collect();
    Ok(DeviceArray::from_host(pool, &coef_f))
}

/// Soft-threshold `sign(t)·max(|t| − l1_reg, 0)` (the un-normalized CD numerator;
/// the caller divides by `‖X_j‖² + l2_reg`). Exact zero when `|t| ≤ l1_reg`
/// (Pitfall 1 — the sparsity pattern).
fn soft_threshold(t: f64, l1_reg: f64) -> f64 {
    if t > l1_reg {
        t - l1_reg
    } else if t < -l1_reg {
        t + l1_reg
    } else {
        0.0
    }
}

/// Validate the CD operand geometry (ASVS V5 / T-05-05-01): `x` must be the
/// `n × d` row-major design (`x.len() == n*d`) and `y` the length-`n` target,
/// both checked BEFORE any unsafe launch.
fn validate_geometry(x_len: usize, y_len: usize, n: usize, d: usize) -> Result<(), PrimError> {
    if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "cd.x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if y_len != n {
        return Err(PrimError::ShapeMismatch {
            operand: "cd.y",
            rows: n,
            cols: 1,
            len: y_len,
        });
    }
    Ok(())
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side scalar math.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cd is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn from_f64<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cd is f32/f64 only"),
    }
}
