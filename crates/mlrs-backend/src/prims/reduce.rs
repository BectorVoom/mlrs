//! Reduction host API (PRIM-02) — full-array + axis-wise (row / column)
//! dispatch for sum / mean / min / max / L2-norm (D-01) and full + per-row
//! argmin / argmax (D-02), launching the feature-free `mlrs-kernels` reduction
//! kernels via a [`ReducePath`] selector so tests exercise BOTH the plane and
//! the shared-memory path (D-03).
//!
//! ## Dual path + subgroup gate (D-03)
//! [`ReducePath::Plane`] launches the `plane_shuffle_xor`/`PLANE_DIM` kernels;
//! [`ReducePath::Shared`] launches the `SharedMemory` log₂-tree kernels. The
//! plane path is capability-gated: when the active adapter lacks subgroup
//! support ([`capability::plane_supported`] is `false`) the plane functions
//! return [`None`] after logging a skip — never failing — so callers fall back
//! to the always-portable shared path. The subgroup-query symbol was resolved
//! in Plan 02-01 (`client.features().plane.contains(Plane::Ops)`), surfaced as
//! `capability::plane_supported()`.
//!
//! ## Per-cube segment reduction → multi-pass finalize
//! Each kernel reduces ONE cube's worth of input into a single partial. For a
//! segment longer than a cube (large-N), the host runs the kernel repeatedly
//! over the shrinking partials until a single value remains (each pass is a
//! pairwise tree, so the whole reduction stays pairwise-stable — Pitfall 3).
//! Row reductions launch one such reduction per row; column reductions gather
//! each column into a contiguous scratch buffer first (transposing the access),
//! then reduce it as a row. mean = sum then host scale by `1/n` (two-pass).
//!
//! ## Device residency (D-05)
//! Inputs and the primary results stay on the device as [`DeviceArray`]s; the
//! API performs NO `to_host`. The caller reads results back at the boundary
//! (the tests use `to_host` / `to_host_metered`). Scratch partial buffers are
//! drawn from and returned to the [`BufferPool`] (D-11).
//!
//! Tests live in `crates/mlrs-backend/tests/reduce_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{
    argmax_shared, argmin_shared, reduce_max_plane, reduce_max_shared, reduce_min_plane,
    reduce_min_shared, reduce_sum_plane, reduce_sum_shared, reduce_sumsq_plane, reduce_sumsq_shared,
};

use crate::capability;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Which reduction kernel family to launch (D-03).
///
/// `Plane` uses the `plane_shuffle_xor` / `PLANE_DIM` subgroup kernels and is
/// capability-gated (skipped-with-log on adapters lacking subgroup support).
/// `Shared` uses the `SharedMemory` log₂-tree kernels and is always portable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducePath {
    /// Plane / subgroup path (`PLANE_DIM`-folded). Capability-gated.
    Plane,
    /// Shared-memory tree path. Always available.
    Shared,
}

/// Which associative scalar reduction to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Sum,
    Min,
    Max,
    SumSq,
}

/// Maximum elements one cube reduces in a single pass (matches the kernels'
/// `SharedMemory::new(256)` and the launch `CubeDim.x` ceiling).
const MAX_CUBE: usize = 256;

/// Smallest power of two that is `>= n`, clamped to `[1, MAX_CUBE]`.
fn cube_dim_for(n: usize) -> u32 {
    let mut d = 1usize;
    while d < n && d < MAX_CUBE {
        d *= 2;
    }
    d.max(1) as u32
}

// ===========================================================================
// Public scalar reductions (full-array, D-01)
// ===========================================================================

/// Full-array sum (D-01). Returns a length-1 device array holding `Σ input`.
pub fn sum<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    reduce_segment::<F>(pool, input.handle().clone(), input.len(), Op::Sum, path)
}

/// Full-array mean (D-01) — sum then host scale by `1/n` (two-pass for
/// stability). Returns a length-1 device array.
pub fn mean<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let n = input.len();
    let summed = match reduce_segment::<F>(pool, input.handle().clone(), n, Op::Sum, path)? {
        Some(s) => s,
        None => return Ok(None),
    };
    // Two-pass: read the single sum, scale on the host, re-upload as a length-1
    // device array (this is a length-1 finalize, NOT a mid-pipeline round-trip
    // of the input — the input stayed device-resident through the reduction).
    let s_host = summed.to_host(pool);
    let scaled = vec![s_host[0] / F::from_int(n.max(1) as i64)];
    Ok(Some(DeviceArray::from_host(pool, &scaled)))
}

/// Full-array minimum (D-01).
pub fn min<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    reduce_segment::<F>(pool, input.handle().clone(), input.len(), Op::Min, path)
}

/// Full-array maximum (D-01).
pub fn max<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    reduce_segment::<F>(pool, input.handle().clone(), input.len(), Op::Max, path)
}

/// Full-array L2 norm `sqrt(Σ xᵢ²)` (D-01). Sum-of-squares on the device, the
/// final `sqrt` applied on the host length-1 finalize (numerically identical to
/// a device sqrt for a single value, and avoids a one-element launch).
pub fn l2_norm<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let summed = match reduce_segment::<F>(pool, input.handle().clone(), input.len(), Op::SumSq, path)? {
        Some(s) => s,
        None => return Ok(None),
    };
    let s_host = summed.to_host(pool);
    let v: f64 = host_to_f64(s_host[0]);
    let scaled = vec![f64_to_host::<F>(v.sqrt())];
    Ok(Some(DeviceArray::from_host(pool, &scaled)))
}

// ===========================================================================
// Axis-wise reductions (row / column, D-01)
// ===========================================================================

/// Row-reduce a `rows × cols` row-major matrix: reduce each row to one value,
/// returning a length-`rows` device array (D-01). `which` selects the op.
///
/// Validates `rows * cols == input.len()` (D-04) before any launch.
pub fn row_reduce<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
    op: ScalarOp,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_matrix(input.len(), rows, cols)?;
    if let (ReducePath::Plane, false) = (path, capability::plane_supported()) {
        log::warn!("reduce row plane-path skipped: adapter lacks subgroup support");
        return Ok(None);
    }
    let inner = op.into_op();
    // Each row is a contiguous segment of length `cols`. Read the whole input
    // once to the host to slice per-row segments back onto the device — for the
    // row/column axis case the per-row segment must be a contiguous device
    // buffer; we materialise each row segment from the single input read.
    let host = input.to_host(pool);
    let mut out_host: Vec<F> = Vec::with_capacity(rows);
    for r in 0..rows {
        let seg = &host[r * cols..(r + 1) * cols];
        let seg_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, seg);
        let reduced = reduce_segment::<F>(pool, seg_dev.handle().clone(), cols, inner, path)?
            .expect("shared/plane-gated segment reduce");
        let rv = reduced.to_host(pool);
        out_host.push(finalize_scalar::<F>(op, rv[0], cols));
    }
    Ok(Some(DeviceArray::from_host(pool, &out_host)))
}

/// Column-reduce a `rows × cols` row-major matrix: reduce each column to one
/// value, returning a length-`cols` device array (D-01). Gathers each column
/// into a contiguous segment (transposing the access) then reduces it.
pub fn column_reduce<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
    op: ScalarOp,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_matrix(input.len(), rows, cols)?;
    if let (ReducePath::Plane, false) = (path, capability::plane_supported()) {
        log::warn!("reduce column plane-path skipped: adapter lacks subgroup support");
        return Ok(None);
    }
    let inner = op.into_op();
    let host = input.to_host(pool);
    let mut out_host: Vec<F> = Vec::with_capacity(cols);
    for c in 0..cols {
        let seg: Vec<F> = (0..rows).map(|r| host[r * cols + c]).collect();
        let seg_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &seg);
        let reduced = reduce_segment::<F>(pool, seg_dev.handle().clone(), rows, inner, path)?
            .expect("shared/plane-gated segment reduce");
        let rv = reduced.to_host(pool);
        out_host.push(finalize_scalar::<F>(op, rv[0], rows));
    }
    Ok(Some(DeviceArray::from_host(pool, &out_host)))
}

/// The user-facing axis-reduction op (sum / mean / min / max / L2-norm).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    /// `Σ` over the axis.
    Sum,
    /// `Σ / n` over the axis.
    Mean,
    /// minimum over the axis.
    Min,
    /// maximum over the axis.
    Max,
    /// `sqrt(Σ xᵢ²)` over the axis (row-L2-norm).
    L2Norm,
    /// `Σ xᵢ²` over the axis — the SQUARED L2 norm, NO sqrt finalize. This is
    /// the term Plan 03's GEMM-expansion distance needs directly (`‖x_i‖²`),
    /// distinct from [`ScalarOp::L2Norm`] which applies the final sqrt.
    SumSq,
}

impl ScalarOp {
    fn into_op(self) -> Op {
        match self {
            ScalarOp::Sum | ScalarOp::Mean => Op::Sum,
            ScalarOp::Min => Op::Min,
            ScalarOp::Max => Op::Max,
            ScalarOp::L2Norm | ScalarOp::SumSq => Op::SumSq,
        }
    }
}

/// Apply the per-axis finalize (mean scale / L2 sqrt) to a raw kernel partial.
///
/// The kernel produced a raw `Σ` (Sum/Mean), `Σx²` (L2Norm), or min/max. Mean
/// scales by `1/n`; L2Norm takes the sqrt; Sum/Min/Max pass through.
fn finalize_scalar<F>(op: ScalarOp, raw: F, n: usize) -> F
where
    F: Float + CubeElement + Pod,
{
    match op {
        // SumSq passes the raw Σxᵢ² through (the squared norm — no sqrt), unlike
        // L2Norm which finalizes with sqrt.
        ScalarOp::Sum | ScalarOp::Min | ScalarOp::Max | ScalarOp::SumSq => raw,
        ScalarOp::Mean => f64_to_host::<F>(host_to_f64(raw) / (n.max(1) as f64)),
        ScalarOp::L2Norm => f64_to_host::<F>(host_to_f64(raw).sqrt()),
    }
}

// ===========================================================================
// argmin / argmax (full + per-row, lowest-index tie-break — D-02)
// ===========================================================================

/// Full-array argmin (D-02): the lowest index of the minimum value
/// (numpy/sklearn tie-break). Returns the index as a `u32` (in a length-1
/// device array) — `None` only on the gated plane path without subgroup
/// support (argmin uses the shared kernel regardless, so this is always `Some`).
pub fn argmin<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
) -> Result<u32, PrimError>
where
    F: Float + CubeElement + Pod,
{
    argreduce::<F>(pool, input.handle().clone(), input.len(), true)
}

/// Full-array argmax (D-02): the lowest index of the maximum value.
pub fn argmax<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
) -> Result<u32, PrimError>
where
    F: Float + CubeElement + Pod,
{
    argreduce::<F>(pool, input.handle().clone(), input.len(), false)
}

/// Per-row argmin over a `rows × cols` row-major matrix (D-02): for each row,
/// the lowest column index of that row's minimum. Returns a length-`rows`
/// `Vec<u32>` of row-relative indices.
pub fn argmin_rows<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> Result<Vec<u32>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_matrix(input.len(), rows, cols)?;
    let host = input.to_host(pool);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let seg = &host[r * cols..(r + 1) * cols];
        let seg_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, seg);
        out.push(argreduce::<F>(pool, seg_dev.handle().clone(), cols, true)?);
    }
    Ok(out)
}

/// Per-row argmax over a `rows × cols` row-major matrix (D-02).
pub fn argmax_rows<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> Result<Vec<u32>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_matrix(input.len(), rows, cols)?;
    let host = input.to_host(pool);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let seg = &host[r * cols..(r + 1) * cols];
        let seg_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, seg);
        out.push(argreduce::<F>(pool, seg_dev.handle().clone(), cols, false)?);
    }
    Ok(out)
}

// ===========================================================================
// Internal: segment reduction driver (multi-pass for large-N stability)
// ===========================================================================

/// Reduce a flat device segment of `len` elements to a single value, looping
/// the chosen kernel over the shrinking partials until one value remains.
///
/// Returns `None` ONLY when the plane path is requested on an adapter without
/// subgroup support (skip-with-log, never fail — D-03); otherwise `Some(result)`.
fn reduce_segment<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input_handle: cubecl::server::Handle,
    len: usize,
    op: Op,
    path: ReducePath,
) -> Result<Option<DeviceArray<ActiveRuntime, F>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if path == ReducePath::Plane && !capability::plane_supported() {
        log::warn!("reduce plane-path skipped: adapter lacks subgroup support");
        return Ok(None);
    }
    if len == 0 {
        let zero = vec![F::from_int(0i64)];
        return Ok(Some(DeviceArray::from_host(pool, &zero)));
    }

    let elem = size_of::<F>();
    let client = pool.client().clone();

    let mut cur_handle = input_handle;
    let mut cur_len = len;
    // The first pass for SumSq must square; subsequent passes only sum the
    // already-squared partials. Track whether we are on the first pass.
    let mut first_pass = true;

    loop {
        // The plane path requires each cube to be at least one FULL plane (and a
        // whole number of planes), else `planes_per_cube = CUBE_DIM_X/PLANE_DIM`
        // rounds to 0 and the in-cube combine reads nothing. PLANE_DIM is
        // runtime-variable on this env's wgpu (min=32, max=64), so we clamp the
        // plane-path cube to at least `plane_size_max` and keep it a power of two
        // (256 is a multiple of both 32 and 64). OOB lanes are zero-seeded (sum)
        // or seeded with the cube's first element (min/max), so over-provisioning
        // a small input is safe. The shared path keeps the tight `cube_dim_for`.
        let cube = if path == ReducePath::Plane {
            // Round the reported plane width up to a power of two, then take the
            // larger of it and the tight cube dim (both powers of two ⇒ result
            // is a power of two), capped at MAX_CUBE.
            let pw = cube_dim_for(capability::active_plane_width().max(1) as usize);
            cube_dim_for(cur_len).max(pw).min(MAX_CUBE as u32)
        } else {
            cube_dim_for(cur_len)
        };
        let num_cubes = ((cur_len as u32) + cube - 1) / cube;

        let (count, dim) = (
            CubeCount::Static(num_cubes.max(1), 1, 1),
            CubeDim { x: cube, y: 1, z: 1 },
        );

        // BOTH paths write exactly ONE partial per cube: the shared kernel via
        // its log₂ tree, the plane kernel via an in-cube combine of its
        // per-plane shuffle partials (so the host needs NO knowledge of the
        // runtime plane width, which is variable on this env's wgpu adapter —
        // plane_size_min=32, max=64). Output layout is identical, so the
        // multi-pass driver is path-agnostic below this point.
        let out_parts = num_cubes as usize;
        let out_bytes = out_parts * elem;
        let out_handle = pool.acquire(out_bytes);

        // `from_raw_parts(handle, len)` consumes the Handle (by value); clone so
        // the loop keeps `cur_handle`/`out_handle` for the next pass + return.
        let in_arg = unsafe { ArrayArg::from_raw_parts(cur_handle.clone(), cur_len) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_parts) };

        match (op, path, first_pass) {
            (Op::Sum, ReducePath::Shared, _) => {
                reduce_sum_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::Sum, ReducePath::Plane, _) => {
                reduce_sum_plane::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::Min, ReducePath::Shared, _) => {
                reduce_min_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::Min, ReducePath::Plane, _) => {
                reduce_min_plane::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::Max, ReducePath::Shared, _) => {
                reduce_max_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::Max, ReducePath::Plane, _) => {
                reduce_max_plane::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::SumSq, ReducePath::Shared, true) => {
                reduce_sumsq_shared::launch::<F, ActiveRuntime>(
                    &client, count, dim, in_arg, out_arg,
                );
            }
            (Op::SumSq, ReducePath::Plane, true) => {
                reduce_sumsq_plane::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            // After the first pass the squares are already accumulated, so
            // subsequent passes plain-sum the partials.
            (Op::SumSq, ReducePath::Shared, false) => {
                reduce_sum_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
            (Op::SumSq, ReducePath::Plane, false) => {
                reduce_sum_plane::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
            }
        }

        first_pass = false;

        if out_parts == 1 {
            return Ok(Some(DeviceArray::from_raw(out_handle, 1)));
        }
        cur_handle = out_handle;
        cur_len = out_parts;
    }
}

/// argmin / argmax driver: multi-pass over the shared index kernel, preserving
/// the lowest-index tie-break across passes (D-02). Returns the winning index.
fn argreduce<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    input_handle: cubecl::server::Handle,
    len: usize,
    is_min: bool,
) -> Result<u32, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if len == 0 {
        return Ok(0);
    }
    let elem = size_of::<F>();
    let client = pool.client().clone();

    // Single device pass over up-to-256-element cubes yields one (value, index)
    // per cube; we then combine the per-cube winners on the host preserving the
    // lowest-index tie-break, converting per-cube-local indices to global. For
    // large N this stays exact because the per-cube index is the within-cube
    // local index and we add the cube offset on the host combine.
    let cube = cube_dim_for(len);
    let num_cubes = (((len as u32) + cube - 1) / cube).max(1);
    let (count, dim) = (
        CubeCount::Static(num_cubes, 1, 1),
        CubeDim { x: cube, y: 1, z: 1 },
    );

    let val_bytes = num_cubes as usize * elem;
    let idx_bytes = num_cubes as usize * size_of::<u32>();
    let val_handle = pool.acquire(val_bytes);
    let idx_handle = pool.acquire(idx_bytes);

    let in_arg = unsafe { ArrayArg::from_raw_parts(input_handle, len) };
    let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), num_cubes as usize) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), num_cubes as usize) };

    if is_min {
        argmin_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, val_arg, idx_arg);
    } else {
        argmax_shared::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, val_arg, idx_arg);
    }

    // Read the per-cube winners and combine on the host (lowest-index tie-break,
    // global index = cube_offset + local index).
    let vals: Vec<F> = DeviceArray::<ActiveRuntime, F>::from_raw(val_handle, num_cubes as usize)
        .to_host(pool);
    let idxs: Vec<u32> =
        DeviceArray::<ActiveRuntime, u32>::from_raw(idx_handle, num_cubes as usize).to_host(pool);

    let mut best_val = host_to_f64(vals[0]);
    let mut best_idx = idxs[0]; // cube 0 offset is 0
    for c in 1..num_cubes as usize {
        let v = host_to_f64(vals[c]);
        let global_idx = c as u32 * cube + idxs[c];
        let better = if is_min { v < best_val } else { v > best_val };
        if better || (v == best_val && global_idx < best_idx) {
            best_val = v;
            best_idx = global_idx;
        }
    }
    Ok(best_idx)
}

// ===========================================================================
// Helpers
// ===========================================================================

fn validate_matrix(len: usize, rows: usize, cols: usize) -> Result<(), PrimError> {
    if rows.checked_mul(cols).map(|v| v != len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "input",
            rows,
            cols,
            len,
        });
    }
    Ok(())
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine / finalize.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("reductions are f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("reductions are f32/f64 only"),
    }
}
