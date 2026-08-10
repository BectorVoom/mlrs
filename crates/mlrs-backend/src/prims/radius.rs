//! `radius` — the DEVICE arm of `radius_neighbors` (NEIGH-RADIUS-GPU).
//!
//! Turns a device-resident `rows × n_train` distance tile into the ragged match
//! set (`d <= radius`) with two kernel launches and ONE small readback, in place
//! of dragging the whole tile back to the host and scanning it there.
//!
//! ## What this removes
//! The tile is `rows · n_train` elements — at the shipping bench rung
//! (`n_query = 2_000`, `n_train = 20_000`, f32) that is 160 MB per query, moved
//! across the bus to be looked at once. The match set at a useful radius is a
//! few percent of it. This arm moves instead:
//!
//! - `rows · segs` `u32` counts back (the count pass's output),
//! - the same-sized offset vector up,
//! - the matches themselves back.
//!
//! ## Why the host still computes the prefix sum
//! An exclusive prefix sum over `rows · segs` (~256 K u32 at the rung above) is
//! a sequential scan the host does in microseconds, and doing it host-side is
//! also what lets this arm allocate the output buffers at their EXACT total size
//! — the count is not known before the first pass, and a device-side scan would
//! still have to report that total back before anything could be allocated. The
//! same readback therefore serves both purposes.
//!
//! ## Ordering
//! [`radius_compact_segments`] emits each query row's matches in ascending
//! TRAINING INDEX order (see that kernel's docs for why segment ownership plus a
//! row-major prefix sum is sufficient), which is the order sklearn's brute-force
//! `radius_neighbors(sort_results=False)` returns and the order the oracle
//! compares positionally.
//!
//! Tests live in `crates/mlrs-backend/tests/radius_scan_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{ArrayArg, CubeElement, Float};

use mlrs_core::{f64_to_host, PrimError};
use mlrs_kernels::radius::{radius_compact_segments, radius_count_segments};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// The ragged match set of one `radius_neighbors` scan, in the CSR-without-
/// `indptr` layout both arms return: per-row matches CONCATENATED, plus each
/// row's count so the caller can derive the offsets with a running sum.
///
/// Shared by the device arm here and the host arm ([`radius_host`](super::radius_host))
/// so the dispatch site handles one type and the two arms cannot drift on the
/// layout contract.
#[derive(Debug)]
pub struct RadiusMatches<F> {
    /// Per-match TRUE metric distances (boundary transform already applied),
    /// row-major concatenated.
    pub distances: Vec<F>,
    /// Per-match training-point indices, row-major concatenated, ASCENDING
    /// within each row.
    pub indices: Vec<i32>,
    /// Per-row match count; `distances`/`indices` split back into rows by its
    /// running sum.
    pub counts: Vec<u32>,
}

/// Training columns one unit scans in the count/compaction passes.
///
/// The row is cut into `ceil(cols / SEGMENT_COLS)` segments so the passes have
/// `rows · segs` units of work rather than `rows` — a 2_000-row tile against
/// 20_000 training points is 2_000 units with one-unit-per-row and 160_000 with
/// this, which is the difference between leaving a GPU idle and filling it.
///
/// Smaller segments also shrink the readback (`rows · segs` counts), so this is
/// a two-sided trade rather than "smaller is better": 128 keeps the count vector
/// at ~1 MB for the rung above while still over-subscribing every backend's unit
/// count by a wide margin.
const SEGMENT_COLS: usize = 128;

/// Should the DEVICE arm serve this scan?
///
/// **Off by default on every backend** — it is opt-in through
/// `MLRS_RADIUS_DEVICE=1`. That is a measurement, not a preference: on both GPU
/// backends this machine has, the fused HOST scan
/// ([`radius_host`](super::radius_host)) is 3-6x faster than this arm and is the
/// only one that beats sklearn at all. Wall clock, `n_train = 20_000`,
/// `n_query = 2_000`, `d = 16`, f32, euclidean, best of 3 interleaved
/// (`scripts/bench_nearest_neighbors_params.py radius`), as the SPEEDUP over
/// sklearn's `algorithm='brute'`:
///
/// | backend | density | this arm | tile readback | host arm |
/// |---------|---------|----------|---------------|----------|
/// | rocm    | 1%      | 0.33x    | 0.27x         | **2.19x**|
/// | rocm    | 15%     | 0.47x    | 0.44x         | **1.80x**|
/// | rocm    | 60%     | 0.71x    | 0.67x         | **2.23x**|
/// | wgpu    | 1%      | 0.23x    | 0.43x         | **1.52x**|
/// | wgpu    | 15%     | 0.39x    | 0.43x         | **1.61x**|
/// | wgpu    | 60%     | 0.49x    | 0.80x         | **1.89x**|
///
/// The rocm device is an INTEGRATED Radeon 860M, where the readback this arm
/// removes is a memcpy over memory the CPU already owns — so it removes almost
/// nothing and pays two kernel launches plus a count round-trip for it. On wgpu
/// it is worse than the readback it replaces, the branchy-kernel launch cost
/// that backend charges (the same pathology `kmeans`' Elkan arm documents).
///
/// It is kept, tested and shipped because the configuration it was BUILT for is
/// not testable here: a discrete GPU across PCIe, where the tile is a genuine
/// `rows × n_train` bus transfer (160 MB at the rung above) rather than a
/// memcpy. Nothing in the numbers above speaks to that case — do not read this
/// table as "the device arm is slow", read it as "on a shared-memory device
/// there was nothing to save". Turn it on there and measure before believing
/// either answer.
pub fn radius_device_applicable(rows: usize, cols: usize) -> bool {
    if !(rows > 0 && cols > 0) {
        return false;
    }
    // Opt-in only; unset means the host arm serves the scan (module docs).
    crate::abflag::var("MLRS_RADIUS_DEVICE")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// Threshold + compact one device-resident distance tile into its matches.
///
/// `dist` is the `rows × cols` row-major tile as `metric_distance` produced it,
/// `thresh` the radius already expressed in the TILE's units (`radius²` when
/// `needs_sqrt`, `radius` otherwise) and `needs_sqrt` that same flag, which roots
/// the kept values. `train_base` is added to every emitted index (0 unless the
/// caller is tiling the training axis).
///
/// The tile is left untouched — the caller owns it and releases it.
pub fn radius_scan_device_tile<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dist: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
    thresh: f64,
    needs_sqrt: bool,
    train_base: usize,
) -> Result<RadiusMatches<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: every bound the kernels index with, BEFORE any unsafe launch. ---
    if rows
        .checked_mul(cols)
        .map(|v| v != dist.len())
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "dist",
            rows,
            cols,
            len: dist.len(),
        });
    }
    for (operand, dim) in [("rows", rows), ("cols", cols), ("train_base", train_base)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let segs = cols.div_ceil(SEGMENT_COLS).max(1);
    let seg_len = cols.div_ceil(segs).max(1);
    // Re-derive the segment count from the length actually used, so the two
    // kernels' `segs`/`seg_len` pair tiles `0..cols` exactly once even when the
    // ceiling division above rounded the length up (e.g. cols = 300 → segs 3,
    // seg_len 100; cols = 257 → segs 3, seg_len 86, and 3·86 = 258 > 257, whose
    // last segment the kernels clamp).
    let segs = cols.div_ceil(seg_len).max(1);
    let n_seg = rows.checked_mul(segs).ok_or(PrimError::Overflow {
        operand: "radius_segments",
        lhs: rows,
        rhs: segs,
    })?;

    let thresh_f: F = f64_to_host::<F>(thresh);
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(n_seg, crate::capability::gather_launch_width());
    // `CubeCount` is not `Copy`, and the compaction pass launches the SAME grid.
    let count2 = count.clone();

    // --- 1. COUNT pass: matches per segment. ---
    let counts_handle = pool.acquire(n_seg * size_of::<u32>());
    // SAFETY: lengths are the validated element counts; the kernel bounds-checks
    // its flattened position against `rows*segs` and clamps its column range to
    // `cols`, and each unit writes only its own count slot.
    let dist_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), dist.len()) };
    let counts_arg = unsafe { ArrayArg::from_raw_parts(counts_handle.clone(), n_seg) };
    radius_count_segments::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        dist_arg,
        counts_arg,
        rows as u32,
        cols as u32,
        segs as u32,
        seg_len as u32,
        thresh_f,
    );

    // --- 2. The ONE readback: the per-segment counts, from which the host
    //        derives both the exclusive offsets and the exact output size. ---
    let counts_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(counts_handle, n_seg);
    let seg_counts: Vec<u32> = counts_dev.to_host(pool);
    counts_dev.release_into(pool);

    let mut offsets: Vec<u32> = Vec::with_capacity(n_seg);
    let mut total: usize = 0;
    for &c in &seg_counts {
        offsets.push(total as u32);
        total += c as usize;
    }
    // Per-ROW counts are the segment counts summed within a row — the caller's
    // half of the CSR layout.
    let mut row_counts: Vec<u32> = Vec::with_capacity(rows);
    for r in 0..rows {
        row_counts.push(seg_counts[r * segs..(r + 1) * segs].iter().sum());
    }
    if total > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "radius_matches",
            rows: total,
            cols: 0,
            len: u32::MAX as usize,
        });
    }

    if total == 0 {
        return Ok(RadiusMatches {
            distances: Vec::new(),
            indices: Vec::new(),
            counts: row_counts,
        });
    }

    // --- 3. COMPACT pass: rescan the (still device-resident) tile, writing each
    //        segment's matches at its offset. ---
    let offsets_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, &offsets);
    let val_handle = pool.acquire(total * size_of::<F>());
    let idx_handle = pool.acquire(total * size_of::<u32>());
    // SAFETY: as above, plus: `offsets` is the exclusive prefix sum of the counts
    // the FIRST pass wrote over the SAME tile and the SAME threshold, so a
    // segment writes at most `counts[g]` elements starting at `offsets[g]`, all
    // within `total`.
    let dist_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), dist.len()) };
    let off_arg = unsafe { ArrayArg::from_raw_parts(offsets_dev.handle().clone(), n_seg) };
    let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), total) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), total) };
    radius_compact_segments::launch::<F, ActiveRuntime>(
        &client,
        count2,
        dim,
        dist_arg,
        off_arg,
        val_arg,
        idx_arg,
        rows as u32,
        cols as u32,
        segs as u32,
        seg_len as u32,
        thresh_f,
        u32::from(needs_sqrt),
        train_base as u32,
    );

    let val_dev = DeviceArray::<ActiveRuntime, F>::from_raw(val_handle, total);
    let idx_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(idx_handle, total);
    let distances: Vec<F> = val_dev.to_host(pool);
    let indices_u32: Vec<u32> = idx_dev.to_host(pool);
    val_dev.release_into(pool);
    idx_dev.release_into(pool);
    offsets_dev.release_into(pool);

    // u32 → i32 (D-06): the values are training indices in `[0, n_train)`, so
    // the cast is exact.
    let indices: Vec<i32> = indices_u32.into_iter().map(|u| u as i32).collect();

    Ok(RadiusMatches {
        distances,
        indices,
        counts: row_counts,
    })
}
