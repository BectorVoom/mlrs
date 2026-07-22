//! `colmean` — multi-pass shared-memory column-sum tree-reduction kernel
//! (`prims::center::center_columns` perf lever, D-05).
//!
//! Feature-free `#[cube]` kernel generic over `<F: Float + CubeElement>`,
//! composed by `mlrs_backend::prims::center`.
//!
//! ## Why this exists (found via Kaggle T4 CUDA profiling of Ridge/
//! ## LinearRegression's large-`n_samples` paths)
//! `center_columns` computed its column mean via the generic
//! `prims::reduce::column_reduce`, which — despite its module docs claiming
//! "device-resident, no host round-trip" — actually does the OPPOSITE: it
//! reads the WHOLE `n × d` input to host ONCE, then LOOPS over each of the
//! `d` columns doing a fresh host→device upload (`DeviceArray::from_host` of
//! a length-`n` gathered column), a kernel launch, and a BLOCKING
//! device→host readback of the single reduced scalar — `d` fully-synchronous
//! round-trips per call. On a same-process integrated adapter (wgpu on a
//! laptop iGPU) the sync/transfer latency is negligible and this is masked;
//! on a real discrete GPU across PCIe (a Kaggle T4) each round-trip pays real
//! latency, and at `d` in the tens this dominates wall-clock fit time (Ridge
//! `RIDGE_PROFILE`/LinearRegression `LR_PROFILE` T4 runs both showed `center`
//! costing MORE than `gram_xty`/`eig`/`solve` combined, and losing outright to
//! sklearn CPU at `n=1_000_000`). This is the exact "iterative prims are
//! host-sync-bound" pathology the project has hit before (KMeans' GEMM sums,
//! LinearRegression's original Gram GEMM) applied to a THIRD prim family.
//!
//! ## Design: batch `column_reduce`'s OWN multi-pass tree algorithm across
//! ## all `d` columns, instead of running it serially once per column
//! `prims::reduce::reduce_segment` (the engine behind `column_reduce`)
//! already gets its numerical accuracy from a MULTI-PASS `SharedMemory`
//! `log₂`-tree reduction (`reduce_sum_shared`): each pass shrinks the
//! remaining element count by ~256×, repeated until 1 remains — depth
//! `O(log₂ n)`, not `O(n)`. [`column_sum_fold`] is the SAME algorithm
//! (verbatim tree-combine shape), but its cube GRID has a second axis for the
//! column (`CUBE_POS_Y`), so ONE launch folds ALL `d` columns' current
//! segment simultaneously instead of one column per launch. The host driver
//! ([`mlrs_backend::prims::center::column_mean_shared_impl`]) repeats the
//! pass, feeding each pass's output back in as the next pass's input, until
//! the per-column remaining length hits 1 — `O(log₂ n)` kernel launches
//! total (e.g. 2–3 for `n` up to low millions), NOT `d` host round-trips.
//! The FINAL pass divides by `n` in-kernel (`is_final=1`), so the caller's
//! own readback is the only host↔device crossing in the whole call.
//!
//! An earlier version of this kernel used a single-pass row-blocked
//! accumulation (thread-per-column, linear-chain-per-block) instead of this
//! multi-pass tree. It was numerically WEAKER than the `column_reduce` path
//! it replaced: on a near-zero-mean column (heavy floating-point
//! cancellation — the true mean orders of magnitude smaller than the summed
//! operands) the linear chain's f32 rounding pushed the strict abs-AND-rel
//! oracle tolerance (D-09) over its `1e-5` relative bound
//! (`center_test.rs::center_columns_matches_host_ref_f32`, caught before
//! shipping). Reusing `reduce_sum_shared`'s EXACT tree-combine shape (not an
//! approximation of it) restores the same numerical quality as the path it
//! replaces.
//!
//! ## cubecl-cpu MLIR safety
//! Like `gram.rs`'s `gram_xty_shared`, this kernel only ever touches
//! `F`/`u32` accumulators — but the HOST caller (`prims::center`) still gates
//! cpu off entirely (mirrors the `use_shared_sums`/`use_shared_gram`
//! precedent) and falls back to the already-validated `column_reduce`-based
//! path there, which IS proven correct on cpu (just slow — cpu has no PCIe
//! transfer cost, so the per-column round-trips are comparatively cheap
//! there).

use cubecl::prelude::*;

/// Threads per cube (matches `reduce.rs::reduce_sum_shared`'s `256`-slot
/// `SharedMemory` log₂-tree convention — this kernel reuses that EXACT
/// combine shape, batched over a second column axis).
pub const FOLD_TPB: u32 = 256;

/// Per-grid-dimension cube-count cap (`maxComputeWorkgroupsPerDimension`
/// floor across the supported backends — the same `65_535` bound
/// `prims::gram::launch_cubes_64` folds against). The host driver folds the
/// per-pass block count across the X and Z grid axes so it never exceeds this
/// in any single dimension.
pub const MAX_GRID_DIM: u32 = 65_535;

/// One pass of the multi-pass column-sum tree reduction (see the module
/// docs). Cube grid is `(x, d, z)`: the block index
/// `blk = CUBE_POS_Z·CUBE_COUNT_X + CUBE_POS_X` indexes this pass's output
/// slot (folded across the X and Z grid axes so `nblocks` can exceed the
/// ~65535 workgroups-per-dimension limit — the same 2D-fold trick
/// `prims::gram::launch_cubes_64` uses, here across X/Z so the Y axis stays
/// free for the column), and `CUBE_POS_Y = col` indexes the column
/// (`col < d`). The folded grid can OVERSHOOT `nblocks` (when
/// `CUBE_COUNT_X · CUBE_COUNT_Z > nblocks`), so a cube-uniform `blk < nblocks`
/// guard wraps the whole body — `blk` depends only on `CUBE_POS`, never the
/// thread, so every thread in a slack cube takes the same branch and the
/// interior `sync_cube` barriers stay cube-uniform (no partial-barrier UB).
/// Each in-range cube loads up to `FOLD_TPB` elements of column `col`'s
/// current segment (`input[(blk·FOLD_TPB + tid)·d + col]`, zero-padded past
/// `cur_len`) into `SharedMemory`, then combines them via the SAME pairwise
/// `log₂` tree `reduce_sum_shared` uses (its own docs cite this shape as
/// numerically load-bearing — "pairwise-stable"), and writes ONE combined
/// value per `(blk, col)` to `output`. On the pass where the caller
/// determines this is the LAST one (`is_final=1`, i.e. `output` will have
/// length `d`), the combined sum is divided by `n` in-kernel so the result is
/// the mean directly.
#[cube(launch)]
pub fn column_sum_fold<F: Float + CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    d: u32,
    cur_len: u32,
    nblocks: u32,
    is_final: u32,
    n: u32,
) {
    let tid = UNIT_POS_X;
    let col = CUBE_POS_Y;
    // Fold the block index across the X and Z grid axes (Y carries the
    // column) so `nblocks` is never bounded by any single grid dimension's
    // ~65535 cap. `blk` is CUBE-uniform, so the guard below is a safe barrier
    // scope for the whole cube.
    let blk = CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X;
    if blk < nblocks {
        let mut shared = SharedMemory::<F>::new(256usize);

        let idx = blk * CUBE_DIM_X + tid;
        shared[tid as usize] = if idx < cur_len {
            input[(idx * d + col) as usize]
        } else {
            F::new(0.0_f32)
        };
        sync_cube();

        // Derived from the CUBECL BUILTIN `CUBE_DIM_X` (a runtime register),
        // NOT the Rust `FOLD_TPB` const directly — cubecl's macro
        // constant-folds a `let mut` variable whose initializer is a PURE
        // const expression into an immutable comptime value, and the `s /= 2`
        // below then fails to compile ("Can't have a mutable operation on a
        // const variable"). `CUBE_DIM_X` equals `FOLD_TPB` at launch (the host
        // pins `CubeDim.x = FOLD_TPB`) but is a genuine runtime value to the
        // macro, so `s` stays mutable — the exact pattern
        // `reduce.rs::reduce_sum_shared` already uses.
        let mut s = CUBE_DIM_X / 2u32;
        while s > 0u32 {
            if tid < s {
                let v = shared[(tid + s) as usize];
                shared[tid as usize] += v;
            }
            sync_cube();
            s /= 2u32;
        }
        if tid == 0u32 {
            let mut result = shared[0usize];
            if is_final == 1u32 {
                result = result / F::cast_from(n);
            }
            output[(blk * d + col) as usize] = result;
        }
    }
}
