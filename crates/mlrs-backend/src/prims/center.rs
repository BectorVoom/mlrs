//! Column-mean centering host API — `out[r,c] = a[r,c] - mean(a[:,c])`,
//! device-resident, no host round-trip (D-05). The centering step composes
//! the already-validated `center_columns` (PRIM-03) elementwise kernel; the
//! mean reduction dispatches to the multi-pass shared-memory
//! `column_sum_fold` tree reduction (see "Column-mean perf" below) or, as a
//! fallback, `column_reduce` (PRIM-02).
//!
//! ## Why this exists separately from `covariance.rs` (D-01)
//! `covariance.rs` also centers internally, but only as a step toward the
//! scaled Gram `AᵀA / (n − ddof)`. Callers that need JUST the centered matrix
//! (and its mean, for downstream intercept recovery) — e.g.
//! `LinearRegression`'s large-`n_samples` Gram+eig path (LINEAR-01), which
//! forms an UNSCALED raw Gram itself via `gemm` — would otherwise have to
//! either hand-roll the `column_reduce` + `center_columns` launch dance
//! in the algos layer (violating the primitive-composition convention) or
//! pay `covariance.rs`'s GEMM + scale steps for a Gram they immediately
//! discard. This prim exposes the shared first half on its own.
//!
//! ## Device residency (D-05)
//! Both returned arrays ([`DeviceArray`]s) stay device-resident; this API
//! performs no host read-back.
//!
//! ## Column-mean perf: row-blocked shared-memory, NOT `column_reduce`
//! (found via Kaggle T4 CUDA profiling of `Ridge`/`LinearRegression`'s
//! large-`n_samples` paths, LINEAR-01/02)
//! Despite this module's device-residency claim above, `column_reduce`
//! actually reads the WHOLE input to host once and then does `cols` FURTHER
//! host-device round-trips (one gathered-column upload + kernel launch +
//! blocking scalar readback PER column) — negligible on a same-process
//! integrated adapter, but dominant on a real discrete GPU across PCIe (a T4
//! `RIDGE_PROFILE`/`LR_PROFILE` run showed `center` costing MORE than
//! `gram_xty`/`eig`/`solve` combined, losing outright to sklearn CPU at
//! `n=1_000_000`). [`column_mean`] below dispatches to
//! `mlrs_kernels::colmean::column_sum_fold` — a MULTI-PASS (`O(log₂ rows)`
//! launches, not `O(rows)` host round-trips) shared-memory tree reduction
//! batched over all columns at once (see that module's docs for the full
//! design and the numerical pitfall its single-pass predecessor hit) —
//! falling back to the original `column_reduce` path only on cpu
//! (`SharedMemory` MLIR-unsafe there) or `cols > 4096`.
//!
//! Tests live in `crates/mlrs-backend/tests/center_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::center_columns as center_columns_kernel;
use mlrs_kernels::colmean::{column_sum_fold, FOLD_TPB, MAX_GRID_DIM};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Column-mean-center `a` (`rows × cols`, row-major) on the device, returning
/// `(centered, mean)`: the device-resident centered matrix (`rows × cols`)
/// and the device-resident column-mean vector (length `cols`) used to produce
/// it — a caller that also needs the mean (e.g. for intercept recovery) gets
/// it without a second reduction pass.
///
/// - Shapes are validated (`rows * cols == a.len()`, both dims non-zero)
///   BEFORE any launch; a mismatch returns [`PrimError::ShapeMismatch`].
/// - The mean reduction is an INTERNAL step, not a caller-visible choice (see
///   [`column_mean`]/[`use_shared_colmean`] for the dispatch); it can never be
///   plane-gated to `None` on a non-subgroup adapter (e.g. cpu) — both the
///   fast path and its `column_reduce(.., ReducePath::Shared)` fallback (CR-01
///   precedent from `covariance.rs`) are always-portable.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn center_columns<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (rows, cols): (usize, usize),
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
    validate_geometry(a.len(), (rows, cols))?;

    let mean_dev = column_mean::<F>(pool, a, rows, cols)?;

    let a_len = rows * cols;
    let elem = size_of::<F>();
    let centred_handle = pool.acquire(a_len * elem);
    let client = pool.client().clone();
    let (ccount, cdim) = launch_dims_1d(a_len);
    // SAFETY: `a_len`/`cols` are the carried/validated element counts; the
    // kernel bounds-checks `tid < a.len()` and reads `mean[tid % cols]`
    // (mirrors `covariance.rs`'s identical launch).
    let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), a_len) };
    let mean_arg = unsafe { ArrayArg::from_raw_parts(mean_dev.handle().clone(), cols) };
    let centred_arg = unsafe { ArrayArg::from_raw_parts(centred_handle.clone(), a_len) };
    center_columns_kernel::launch::<F, ActiveRuntime>(
        &client,
        ccount,
        cdim,
        a_arg,
        mean_arg,
        centred_arg,
        cols as u32,
    );
    let centred_dev = DeviceArray::from_raw(centred_handle, a_len);

    Ok((centred_dev, mean_dev))
}

/// Column-mean dispatch (see the module docs): the row-blocked
/// shared-memory fast path ([`use_shared_colmean`]) or the
/// `column_reduce`-based fallback (cpu backend, `cols > 4096`, or the
/// `CENTER_COLUMN_REDUCE` A/B override — mirrors `prims::gram`'s
/// `LR_GRAM_GEMM` precedent).
fn column_mean<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if use_shared_colmean(cols) {
        Ok(column_mean_shared_impl::<F>(pool, a, rows, cols))
    } else {
        Ok(
            column_reduce::<F>(pool, a, rows, cols, ScalarOp::Mean, ReducePath::Shared)?
                .expect("shared path is never plane-gated to None"),
        )
    }
}

/// Whether to use the row-blocked shared-memory column-mean kernel. `false`
/// on the cpu backend (MLIR rejects `SharedMemory` — the `use_shared_gram`
/// precedent in `prims::gram`) and whenever `cols` would exceed the fixed
/// 4096-slot `SharedMemory` budget (`mlrs_kernels::colmean` module docs).
/// `CENTER_COLUMN_REDUCE=1` forces the `column_reduce` fallback everywhere,
/// for A/B benchmarking.
fn use_shared_colmean(cols: usize) -> bool {
    #[cfg(feature = "cpu")]
    {
        let _ = cols;
        false
    }
    #[cfg(not(feature = "cpu"))]
    {
        if std::env::var("CENTER_COLUMN_REDUCE").is_ok() {
            return false;
        }
        cols <= 4096
    }
}

/// Multi-pass column-mean formation — the `center_columns` perf path (see
/// `mlrs_kernels::colmean` module docs for the full numerical rationale).
/// Repeats [`column_sum_fold`], each pass shrinking the per-column remaining
/// length by `FOLD_TPB` (256×), until it reaches 1 — `O(log₂ rows)` kernel
/// launches (e.g. 2-3 for `rows` up to low millions), each a genuine
/// `SharedMemory` `log₂`-tree combine (the SAME numerical shape
/// `reduce.rs::reduce_sum_shared` uses), batched over all `cols` columns per
/// launch so there is NO per-column device round-trip. Each pass's block
/// count is folded across the X/Z grid axes (`MAX_GRID_DIM`) so arbitrarily
/// large `rows` never overflows a single grid dimension. The final pass
/// divides by `rows` in-kernel, so the caller's own readback of the returned
/// length-`cols` mean is the only host↔device crossing in the whole call.
fn column_mean_shared_impl<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    debug_assert!(cols <= 4096, "shared colmean caller must gate cols <= 4096");
    let elem = size_of::<F>();
    let client = pool.client().clone();

    let mut cur_handle = a.handle().clone();
    let mut cur_len = rows;
    let mut cur_is_scratch = false;

    loop {
        let next_len = cur_len.div_ceil(FOLD_TPB as usize).max(1);
        let is_final: u32 = if next_len == 1 { 1 } else { 0 };
        let out_len = next_len * cols;
        let out_handle = pool.acquire(out_len * elem);

        // SAFETY: `cur_len * cols` is the carried/validated element count of
        // `cur_handle` (either `a` on pass 1, or a just-acquired pool buffer
        // on later passes); the kernel bounds-checks `idx < cur_len` and
        // `blk < nblocks`, zero-padding any OOB lane / guarding any slack cube.
        let in_arg = unsafe { ArrayArg::from_raw_parts(cur_handle.clone(), cur_len * cols) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        // Fold the per-pass block count across the X and Z grid axes so it
        // never exceeds `MAX_GRID_DIM` in any single dimension (the Y axis
        // carries the column). Without this, a design with `rows > 65535·256
        // ≈ 16.7M` would request more than `MAX_GRID_DIM` cubes in `grid.x`,
        // which the backend rejects/truncates — silently dropping tail
        // row-blocks and corrupting the column means (and thus the fitted
        // coefficients). Mirrors `prims::gram::launch_cubes_64`'s fold, here
        // across X/Z instead of X/Y. `next_len ≤ ~16.7M`, so `nbz` stays far
        // below `MAX_GRID_DIM` (≤ ~256 even at `rows = 10⁹`).
        let nbx = (next_len as u32).min(MAX_GRID_DIM).max(1);
        let nbz = (next_len as u32).div_ceil(nbx).max(1);
        let cc = CubeCount::Static(nbx, cols.max(1) as u32, nbz);
        let cd = CubeDim { x: FOLD_TPB, y: 1, z: 1 };
        column_sum_fold::launch::<F, ActiveRuntime>(
            &client,
            cc,
            cd,
            in_arg,
            out_arg,
            cols as u32,
            cur_len as u32,
            next_len as u32,
            is_final,
            rows as u32,
        );

        if cur_is_scratch {
            pool.release(cur_handle, cur_len * cols * elem);
        }
        cur_handle = out_handle;
        cur_len = next_len;
        cur_is_scratch = true;

        if cur_len == 1 {
            break;
        }
    }

    DeviceArray::from_raw(cur_handle, cols)
}

/// Validate the center-columns operand geometry. `a` is `rows × cols`;
/// `rows * cols == a.len()`, both dims non-zero (an empty axis has no
/// well-defined mean).
fn validate_geometry(a_len: usize, (rows, cols): (usize, usize)) -> Result<(), PrimError> {
    if rows == 0
        || cols == 0
        || rows.checked_mul(cols).map(|v| v != a_len).unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows,
            cols,
            len: a_len,
        });
    }
    Ok(())
}

/// Ceiling-division per-element launch config, FOLDED across the X/Y grid
/// axes so the cube count never exceeds `MAX_GRID_DIM` in any single
/// dimension. The `center_columns` elementwise kernel addresses its element
/// via the flattened `ABSOLUTE_POS` (which linearizes contiguously across a
/// multi-axis grid: cube `(x, y)` covers elements
/// `[(y·CUBE_COUNT_X + x)·block, +block)`) and bounds-checks `tid < a.len()`,
/// so folding into a 2D grid is transparent to it. The un-folded
/// `Static(cubes, 1, 1)` form (still used by the sibling `covariance.rs` /
/// distance prims) silently exceeds the ~65535 per-dimension cap once
/// `n > 65535·256 ≈ 16.7M` — the large-`n_samples` centering hot path this
/// perf work targets, so it MUST fold here (`center_test.rs`'s ignored
/// grid-fold test covers exactly this size).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    let x = cubes.min(MAX_GRID_DIM).max(1);
    let y = cubes.div_ceil(x).max(1);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
