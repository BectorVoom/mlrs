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
use mlrs_kernels::topk::{select_k, select_k_onepass, select_k_shared};

use crate::capability;
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

    // SAFETY: lengths are the carried/validated element counts (the kernel
    // bounds-checks `row < rows` and only writes `rows * k` slots), NEVER raw
    // caller geometry — mitigates T-05-02-01.
    let dist_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), dist.len()) };
    let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), out_len) };

    if serial_select_forced() || cpu_serial_select() {
        // A/B escape hatch (`MLRS_TOPK_SERIAL=1`): the legacy one-unit-per-row
        // kernel, retained so the parallel path can be measured against it on the
        // actual target device rather than by extrapolation.
        //
        // It is ALSO the cpu default (`cpu_serial_select`) — see that function
        // for why "parallel" and "serial" swap meanings on that backend.
        let (count, dim) = launch_dims_rows(rows);
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
    } else if k <= ONEPASS_K_CAP && !multipass_select_forced() {
        // SINGLE-PASS selection (the default for KNN-sized k): each unit keeps
        // its strided slice's k smallest in a sorted local list, then k head
        // -merge rounds through the shared pair-order tree emit the row's exact
        // ascending top-k. Reads the matrix ONCE where the multi-pass kernel
        // reads it k times — see the kernel docs for the traffic model.
        let (count, dim) = launch_dims_rows_parallel(rows, cols);
        select_k_onepass::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            dist_arg,
            val_arg,
            idx_arg,
            rows as u32,
            cols as u32,
            k as u32,
        );
    } else {
        // One cube per query row (CUBE_POS_X = row) with a POWER-OF-TWO unit
        // width: every unit scans a strided slice of the row and the winners are
        // folded through a shared-memory pair-order tree (bitwise-identical
        // output, `cols / width` less serial work per rank — see the kernel docs
        // for the 41×-of-distance measurement that motivated it). Reached when
        // `k` exceeds the one-pass kernel's local-list capacity, or when forced
        // for A/B via `MLRS_TOPK_MULTIPASS=1`.
        let (count, dim) = launch_dims_rows_parallel(rows, cols);
        select_k_shared::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            dist_arg,
            val_arg,
            idx_arg,
            rows as u32,
            cols as u32,
            k as u32,
        );
    }

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

/// Local-list capacity of `select_k_onepass` (must match the kernel's comptime
/// `Array::new(32)` allocations). Selections with `k` beyond this fall back to
/// the multi-pass `select_k_shared`.
const ONEPASS_K_CAP: usize = 32;

/// Is the legacy serial `select_k` forced via `MLRS_TOPK_SERIAL=1`?
///
/// The parallel kernels are the default. This escape hatch exists ONLY so the
/// paths can be A/B'd on the real target device (a perf kernel must never be
/// gated onto a backend by extrapolating from a different backend's numbers).
fn serial_select_forced() -> bool {
    std::env::var("MLRS_TOPK_SERIAL").map(|v| v == "1").unwrap_or(false)
}

/// Should the cpu backend take the one-unit-per-row `select_k` kernel?
/// DEFAULT YES, and it is the FASTER choice there — the labels "serial" and
/// "parallel" swap meaning on cpu.
///
/// `cubecl-cpu` maps ONE OS THREAD PER UNIT and runs the cube grid as a serial
/// loop inside each thread, so parallelism is `cube_dim` and `sync_cube` is a
/// SPIN barrier across every one of those threads
/// (`cubecl_cpu::compute::compute_task::sync_cube`). Under that model the two
/// shared-memory kernels are not merely slower, they are structurally wrong:
/// `launch_dims_rows_parallel` asks for up to 256 units = 256 OS threads on a
/// machine with a dozen cores, and each of `log₂(width) + 2` barriers per rank
/// makes the runnable threads burn a whole scheduling quantum waiting on
/// descheduled peers.
///
/// The one-unit-per-row kernel inverts that correctly: `CubeDim { x: 1 }` with
/// one cube per row means ONE thread walking every row serially — which is
/// exactly what a cpu wants, since the cube grid is that thread's loop. It uses
/// no `SharedMemory` and no barriers at all (see its kernel docs, which already
/// call out the cubecl-cpu MLIR constraints it was written against).
///
/// This is the same dispatch discipline `prims::knn::cpu_rows_applicable`
/// applies to the fused KNN search, extended to the shared `top_k` prim that
/// `spectral.rs`, `umap.rs` and `knn_graph.rs` also depend on.
/// `MLRS_TOPK_MULTIPASS=1` / `MLRS_TOPK_SERIAL=1` still force their arms for
/// on-target A/B.
fn cpu_serial_select() -> bool {
    capability::active_backend_name() == "cpu" && !multipass_select_forced()
}

/// Is the multi-pass `select_k_shared` forced via `MLRS_TOPK_MULTIPASS=1`?
///
/// The single-pass `select_k_onepass` is the default for `k <= ONEPASS_K_CAP`;
/// this escape hatch selects the k-pass kernel so the two can be A/B'd on the
/// real target device.
fn multipass_select_forced() -> bool {
    std::env::var("MLRS_TOPK_MULTIPASS").map(|v| v == "1").unwrap_or(false)
}

/// Launch config for `select_k_shared`: ONE cube per query row (`CUBE_POS_X` =
/// row) with a POWER-OF-TWO unit width.
///
/// The width is the largest power of two `<= min(256, cols)`: 256 caps it at the
/// kernel's SharedMemory size (matching `reduce.rs`), and clamping to `cols`
/// avoids launching units that would own an empty strided slice on narrow
/// selections. The `log₂` tree reduce requires the power of two; `max(1)` keeps a
/// `cols == 1` selection legal (the tree loop is then a no-op and unit 0 emits its
/// own scan result).
fn launch_dims_rows_parallel(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let capped = cols.min(256) as u32;
    // Largest power of two <= capped (capped >= 1 — validate_geometry pinned
    // `1 <= k <= cols`).
    let width = 1u32 << (31 - capped.max(1).leading_zeros());
    (
        CubeCount::Static((rows as u32).max(1), 1, 1),
        CubeDim { x: width, y: 1, z: 1 },
    )
}

/// Launch config for the legacy serial `select_k`: ONE cube per query row
/// (`CUBE_POS_X` = row), a single-unit cube (only unit 0 selects). The kernel
/// bounds-checks `row < rows`, so `rows.max(1)` cubes is exact.
fn launch_dims_rows(rows: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static((rows as u32).max(1), 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}

/// Standard ceiling-division 1D launch config for the in-place sqrt pass over the
/// `rows × k` returned distances (matches `distance.rs::launch_dims_1d`).
///
/// `sqrt_elem` is a barrier-free `ABSOLUTE_POS_X` GATHER, so the block width is
/// whatever the backend schedules best — see [`capability::gather_launch_width`].
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = capability::gather_launch_width();
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
