//! `radius` — device-side threshold + ORDERED compaction of a distance tile
//! (NEIGH-RADIUS-GPU).
//!
//! Two `#[cube]` kernels that turn a materialized `rows × cols` distance tile
//! into the ragged `radius_neighbors` match set — the training indices (and
//! their distances) satisfying `d <= radius` — WITHOUT reading the tile back to
//! the host. The tile is `rows × n_train` elements; the match set is
//! (at the densities a radius query is useful at) orders of magnitude smaller,
//! so what these kernels remove is the transfer, not the arithmetic.
//!
//! ## Two passes over the tile, because the output size is data-dependent
//! A radius query's match COUNT per query row is unknown until the tile has been
//! scanned, so a single pass has nowhere to write. [`radius_count_segments`]
//! counts, the host exclusive-prefix-sums the (small) count vector into output
//! offsets, and [`radius_compact_segments`] rescans and writes. The tile stays
//! device-resident between the two, so the second pass is a device read, not a
//! recomputation of the distances.
//!
//! ## Why SEGMENTS and not one unit per row (the ordering constraint)
//! sklearn's brute-force `radius_neighbors(sort_results=False)` returns each
//! row's matches in ASCENDING TRAINING INDEX order, and mlrs's oracle compares
//! them positionally — so the compaction must be order-preserving. An atomic
//! bump-allocator per row is the usual GPU compaction and is exactly what cannot
//! be used here: it interleaves by completion order.
//!
//! Instead each row is cut into `segs` CONTIGUOUS segments and segment `(r, s)`
//! is owned by one unit, which writes its matches in ascending `j` at an offset
//! the host computed. Because the offsets are the exclusive prefix sum over the
//! count vector in ROW-MAJOR `(r, s)` order, and the output layout is itself
//! row-major-concatenated, the concatenation of every segment's ascending run is
//! the whole result in ascending order — with no cross-unit synchronization and
//! no atomics (which `cubecl-cpu`'s MLIR lowering does not support anyway).
//!
//! `segs > 1` is what keeps the compaction parallel: with one unit per row a
//! 2_000-row tile has 2_000 units of work regardless of how wide the machine is.
//!
//! ## Threshold on the tile's OWN units (Pitfall 8)
//! `metric_distance` reports `needs_sqrt = true` only for Euclidean, whose GEMM
//! expansion returns the order-preserving SQUARED distance; every other metric's
//! kernel emits its true value. The caller therefore passes `radius²` as
//! `thresh` for Euclidean and `radius` otherwise, and `sqrt_flag = 1` makes the
//! compaction root only the values it actually keeps. `sqrt` is monotone on
//! `[0, ∞)`, so thresholding the squared tile selects exactly the rows
//! `sqrt(d) <= radius` would.
//!
//! Both kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature. Tests live in `crates/mlrs-backend/tests/radius_scan_test.rs`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

/// Per-segment match count over a `rows × cols` row-major distance tile:
/// `counts[r*segs + s] = #{ j in segment s of row r : dist[r, j] <= thresh }`.
///
/// One unit per SEGMENT, addressed through the flattened [`ABSOLUTE_POS`] so the
/// launch may fold its cube count across the grid's X/Y axes
/// (`launch_dims_1d_folded`). Each unit owns one output slot — a GATHER, no
/// atomics and no `SharedMemory`.
///
/// `seg_len` is the ceiling division `cols / segs`, so the last segment of a row
/// is short (possibly empty); its `while j < end` simply does not run and the
/// unit writes `0`. The row's segments therefore tile `0..cols` exactly once,
/// which is what makes the prefix sum over `counts` a valid output offset.
#[cube(launch)]
pub fn radius_count_segments<F: Float + CubeElement>(
    dist: &Array<F>,
    counts: &mut Array<u32>,
    rows: u32,
    cols: u32,
    segs: u32,
    seg_len: u32,
    thresh: F,
) {
    let tid = ABSOLUTE_POS;
    let total = rows * segs;
    if tid < total as usize {
        let g = tid as u32;
        let r = g / segs;
        let s = g % segs;
        let start = s * seg_len;
        let mut end = start + seg_len;
        if end > cols {
            end = cols;
        }
        let base = r * cols;

        // `count_seg` is an explicitly typed u32 accumulator (the cube macro
        // needs the annotation to infer a cross-loop scalar's type).
        let mut count_seg: u32 = 0u32;
        let mut j = start;
        while j < end {
            if dist[(base + j) as usize] <= thresh {
                count_seg += 1u32;
            }
            j += 1u32;
        }
        counts[g as usize] = count_seg;
    }
}

/// Write every match of segment `(r, s)` — its training index and its distance —
/// into the flat output buffers starting at `offsets[r*segs + s]`.
///
/// `offsets` is the EXCLUSIVE prefix sum of [`radius_count_segments`]' output in
/// the same `(r, s)` row-major order, computed host-side; the module docs give
/// the argument for why that makes the concatenated result ascending-by-index
/// within every row without any cross-unit ordering primitive.
///
/// `sqrt_flag = 1` roots each kept value (the Euclidean boundary transform,
/// applied to matches only — never to the whole tile). `train_base` is added to
/// every emitted index so a caller tiling over the QUERY axis (which keeps the
/// full training set) can pass `0`, while one tiling the training axis can
/// re-base its segment's indices into the global training set.
///
/// A unit whose segment has no match writes nothing at all, so the buffers need
/// only hold the exact total the count pass reported.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn radius_compact_segments<F: Float + CubeElement>(
    dist: &Array<F>,
    offsets: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    cols: u32,
    segs: u32,
    seg_len: u32,
    thresh: F,
    sqrt_flag: u32,
    train_base: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = rows * segs;
    if tid < total as usize {
        let g = tid as u32;
        let r = g / segs;
        let s = g % segs;
        let start = s * seg_len;
        let mut end = start + seg_len;
        if end > cols {
            end = cols;
        }
        let base = r * cols;

        let mut pos = offsets[g as usize];
        let mut j = start;
        while j < end {
            let v = dist[(base + j) as usize];
            if v <= thresh {
                let mut kept = v;
                if sqrt_flag == 1u32 {
                    kept = F::sqrt(v);
                }
                out_val[pos as usize] = kept;
                out_idx[pos as usize] = train_base + j;
                pos += 1u32;
            }
            j += 1u32;
        }
    }
}
