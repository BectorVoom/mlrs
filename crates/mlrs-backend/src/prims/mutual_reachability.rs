//! Host-launch wrapper for the HDBSCAN mutual-reachability GATHER kernel
//! (HDBS-01, Phase 15, plan 15-05).
//!
//! The device kernel lives in the feature-free `mlrs-kernels` crate
//! (`mutual_reachability::mutual_reachability`); this layer owns the concrete
//! `ActiveRuntime`, validates geometry HOST-SIDE before the `unsafe` launch
//! (T-15-05-V5), and routes the dense distance block + per-row core distances
//! through the kernel, returning the dense `rows_x × rows_y` mutual-reachability
//! matrix device-resident.
//!
//! This is the dense (Variant-A / cosine) device front-end: the WHOLE distance
//! block is divided by `alpha` in-kernel (`d_ij / alpha`) BEFORE the core-distance
//! max, matching the Variant-A alpha placement (the caller supplies RAW distances
//! and RAW core distances; the kernel does the `/alpha`). The kernel is a
//! per-element GATHER with NO cross-thread state — SharedMemory-free, cpu-MLIR-safe
//! (the chebyshev_dist running-max precedent).
//!
//! Tests live in `crates/mlrs-backend/tests/mutual_reachability_test.rs`
//! (AGENTS.md §2); the VALUE oracle there asserts the MR values incl. a
//! duplicate-point row (R-9), not just non-panic.

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::mutual_reachability_kernel;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// The 2D cube edge length for the GATHER launch (matches the kernel's documented
/// `CubeDim {x:16, y:16}` and the `knn_graph.rs::launch_dims_2d` shape). The kernel
/// bounds-checks `i`/`j` regardless, so the two are decoupled for correctness.
const CUBE_DIM_2D: u32 = 16;

/// Build the dense `rows_x × rows_y` mutual-reachability matrix
/// `out[i*rows_y + j] = max(core[i], core[j], d[i*rows_y + j] / alpha)` on-device
/// from a dense distance block `d` and per-row core distances `core`.
///
/// - `d` is the row-major `rows_x × rows_y` distance block (`rows_x * rows_y`
///   elements). For the dense-cosine path this is the full `n × n` cosine distance
///   matrix.
/// - `core` is the per-row core distance, length `rows_x` (== `rows_y` for the
///   square dense path).
/// - `alpha > 0` is the robust-single-linkage scaling (Variant-A placement: the
///   distance is divided by `alpha` BEFORE the core max). The caller's
///   `HdbscanBuilder::build` rejects `alpha <= 0` BEFORE any fit reaches here.
///
/// Geometry is validated HOST-SIDE BEFORE the `unsafe` launch (T-15-05-V5 / ASVS
/// V5): `d.len() == rows_x * rows_y` (operand `"d"`), `core.len() == rows_x`
/// (operand `"core"`), and a `checked_mul` overflow guard on `rows_x * rows_y`
/// plus `u32`-fit guards on the launch dims (T-15-05-OVF). A violation returns
/// [`PrimError::ShapeMismatch`] (the knn_graph.rs precedent — no numeric-range
/// variant exists). The returned matrix is a caller-owned device array.
pub fn mutual_reachability_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    d: &DeviceArray<ActiveRuntime, F>,
    core: &DeviceArray<ActiveRuntime, F>,
    rows_x: usize,
    rows_y: usize,
    alpha: F,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- T-15-05-V5 / ASVS V5: validate geometry HOST-SIDE before any launch. ---
    // checked_mul overflow guard on rows_x * rows_y (T-15-05-OVF) — also the
    // expected `d` length.
    let out_len = rows_x
        .checked_mul(rows_y)
        .ok_or(PrimError::Overflow {
            operand: "d",
            lhs: rows_x,
            rhs: rows_y,
        })?;
    if d.len() != out_len {
        return Err(PrimError::ShapeMismatch {
            operand: "d",
            rows: rows_x,
            cols: rows_y,
            len: d.len(),
        });
    }
    if core.len() != rows_x {
        return Err(PrimError::ShapeMismatch {
            operand: "core",
            rows: rows_x,
            cols: 1,
            len: core.len(),
        });
    }
    // u32-fit guards on the launch dims (rows_x / rows_y are cast to u32 for the
    // kernel launch).
    for (operand, dim) in [("d", rows_x), ("d", rows_y)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let out_handle = pool.acquire(out_len * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_2d(rows_x, rows_y);

    // SAFETY: lengths are the validated element counts (d/out are rows_x*rows_y,
    // core is rows_x); the kernel bounds-checks i<rows_x && j<rows_y (T-15-05-V5).
    // Scalars pass BY VALUE in cubecl 0.10.
    let d_arg = unsafe { ArrayArg::from_raw_parts(d.handle().clone(), d.len()) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(core.handle().clone(), core.len()) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

    mutual_reachability_kernel::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        d_arg,
        c_arg,
        o_arg,
        rows_x as u32,
        rows_y as u32,
        alpha,
    );

    Ok(DeviceArray::from_raw(out_handle, out_len))
}

/// 2D launch config for the GATHER kernel: one unit per output element `(i, j)`,
/// `i` on `ABSOLUTE_POS_X`, `j` on `ABSOLUTE_POS_Y`. Ceiling-division over a 16×16
/// cube (matches the kernel's documented `CubeDim {x:16, y:16}` and
/// `knn_graph.rs::launch_dims_2d`). NEVER a bare 1D `ABSOLUTE_POS` launch
/// (FINDING 002-A).
fn launch_dims_2d(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let bx = CUBE_DIM_2D;
    let by = CUBE_DIM_2D;
    let cx = ((rows as u32) + bx - 1) / bx;
    let cy = ((cols as u32) + by - 1) / by;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: bx, y: by, z: 1 },
    )
}
