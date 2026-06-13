//! `prims::topk` — host orchestration for the top-k select primitive (PRIM,
//! D-02).
//!
//! The launch wrapper for the new `mlrs_kernels::topk::select_k` partial-select
//! kernel: per query ROW of a `rows × cols` distance matrix it returns the `k`
//! smallest distances (ascending) and their column indices, with a LOWEST-INDEX
//! tie-break. It VALIDATES geometry before any `unsafe` launch (ASVS V5 /
//! T-05-02-01), threads an optional reused `out` buffer (D-11), and returns
//! device-resident `(distances, indices)` per query row (the `prims::distance`
//! precedent). The `u32` neighbor indices are re-uploaded as `i32` by the KNN
//! consumers (D-06, plan 08).
//!
//! ## Squared distance in, optional sqrt at the boundary (Pitfall 8 / D-08)
//! Top-k selection runs on the SQUARED distance (the cheaper, order-preserving
//! form — `argpartition` on `d²` selects the same neighbors as on `d`). The
//! optional `sqrt` is applied ONLY to the returned `k` values per row at the
//! boundary, so KNN gets true Euclidean distances without sqrting the whole
//! `rows × cols` matrix. The indices are unaffected by the monotone sqrt.
//!
//! ## Device residency (D-05)
//! Inputs and the two outputs stay on the device as [`DeviceArray`]s; the
//! caller reads them back at the boundary. The output buffers are acquired from
//! the [`BufferPool`] (or reused from the caller's `out`, D-11).
//!
//! Tests live in `crates/mlrs-backend/tests/topk_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::sqrt_elem;
use mlrs_kernels::topk::select_k;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Select the `k` smallest distances + their column indices per query ROW of the
/// `rows × cols` row-major distance matrix `dist` (D-02), with a LOWEST-INDEX
/// tie-break.
///
/// - `dist` is the `rows × cols` distance matrix (one query per row, one train
///   point per column); pass the SQUARED distance (the order-preserving form,
///   Pitfall 8).
/// - Geometry is validated (`rows * cols == dist.len()`) AND `1 <= k <= cols`
///   BEFORE any launch (T-05-02-01 / ASVS V5); a violation returns
///   [`PrimError::ShapeMismatch`] (the distance.rs precedent — no separate
///   `InvalidK` variant in `PrimError`).
/// - `sqrt = true` applies the Euclidean sqrt to ONLY the returned `k` values per
///   row at the boundary (D-08); the indices are unaffected.
/// - The two `rows × k` results are acquired from `pool` when their `out_*` is
///   `None`, else the supplied buffer is reused (D-11). Both stay device-resident
///   (D-05) — NO host round-trip inside this API.
///
/// Returns `(distances, indices)`: `distances` is `rows × k` (`F`), `indices` is
/// `rows × k` (`u32`, re-uploaded as `i32` by the KNN consumers, D-06).
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
#[allow(clippy::too_many_arguments)]
pub fn top_k<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dist: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
    k: usize,
    sqrt: bool,
    out_val: Option<DeviceArray<ActiveRuntime, F>>,
    out_idx: Option<DeviceArray<ActiveRuntime, u32>>,
) -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, u32>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- T-05-02-01 / ASVS V5: validate geometry + k BEFORE any unsafe launch. ---
    validate_geometry(
        dist.len(),
        (rows, cols),
        k,
        out_val.as_ref().map(DeviceArray::len),
        out_idx.as_ref().map(DeviceArray::len),
    )?;

    let out_len = rows * k;
    let velem = size_of::<F>();
    let ielem = size_of::<u32>();

    // Acquire output buffers from the pool only when the caller did not supply a
    // reusable one (D-11). The caller OWNS the returned buffers — never released
    // here.
    let val_handle = match &out_val {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * velem),
    };
    let idx_handle = match &out_idx {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * ielem),
    };

    let client = pool.client().clone();
    // One cube per query row (CUBE_POS_X = row); a 1-unit cube — only unit 0
    // selects (small-k insertion-select, see the kernel docs).
    let (count, dim) = launch_dims_rows(rows);

    // SAFETY: lengths are the carried/validated element counts (the kernel
    // bounds-checks `row < rows` and only writes `rows * k` slots), NEVER raw
    // caller geometry — mitigates T-05-02-01.
    let dist_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), dist.len()) };
    let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), out_len) };

    select_k::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        dist_arg,
        val_arg,
        idx_arg,
        // Scalar args by value in cubecl 0.10 (no ScalarArg — see distance.rs).
        rows as u32,
        cols as u32,
        k as u32,
    );

    // --- Optional Euclidean sqrt over ONLY the returned k values (D-08 / Pitfall
    //     8). Squared distance selects the same neighbors as Euclidean, so the
    //     sqrt is the monotone boundary applied in place over the `rows × k`
    //     distance buffer (never the whole matrix). Indices are unaffected. ---
    if sqrt {
        let (scount, sdim) = launch_dims_1d(out_len);
        let in_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        let sout_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
    }

    // Both results stay device-resident (D-05); the caller reads them back at the
    // boundary.
    Ok((
        DeviceArray::from_raw(val_handle, out_len),
        DeviceArray::from_raw(idx_handle, out_len),
    ))
}

/// Validate top-k operand geometry + `k` (T-05-02-01 / ASVS V5). `dist` is
/// `rows × cols`; `k` must satisfy `1 <= k <= cols`; the optional outputs (if
/// supplied) must each be `rows × k`. Rejected BEFORE any launch so a wrong
/// shape / bad `k` is a recoverable typed error, not an out-of-bounds device
/// read.
fn validate_geometry(
    dist_len: usize,
    (rows, cols): (usize, usize),
    k: usize,
    out_val_len: Option<usize>,
    out_idx_len: Option<usize>,
) -> Result<(), PrimError> {
    if rows.checked_mul(cols).map(|v| v != dist_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "dist",
            rows,
            cols,
            len: dist_len,
        });
    }
    // `1 <= k <= cols` — a k of 0 selects nothing and a k > cols would read past
    // the row. PrimError has no dedicated InvalidK variant (distance.rs uses
    // ShapeMismatch for all geometry violations), so report it as a `k`-vs-`cols`
    // shape mismatch on a synthetic `"k"` operand.
    if k < 1 || k > cols {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: cols,
        });
    }
    // WR-03: rows, cols, k are cast to u32 for the kernel launch geometry; reject
    // an overflowing dimension BEFORE launch so the cast cannot silently truncate
    // into an out-of-bounds device read.
    for (operand, dim) in [("rows", rows), ("cols", cols), ("k", k)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }
    let expect = rows * k;
    if let Some(o) = out_val_len {
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out_val",
                rows,
                cols: k,
                len: o,
            });
        }
    }
    if let Some(o) = out_idx_len {
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out_idx",
                rows,
                cols: k,
                len: o,
            });
        }
    }
    Ok(())
}

/// Launch config for `select_k`: ONE cube per query row (`CUBE_POS_X` = row), a
/// single-unit cube (only unit 0 selects). The kernel bounds-checks `row < rows`,
/// so `rows.max(1)` cubes is exact.
fn launch_dims_rows(rows: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static((rows as u32).max(1), 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}

/// Standard ceiling-division 1D launch config for the in-place sqrt pass over the
/// `rows × k` returned distances (matches `distance.rs::launch_dims_1d`).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
