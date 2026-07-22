//! `gram` — row-blocked shared-memory Gram/Xty accumulation kernels
//! (LINEAR-01 perf lever, D-02 Gram+eig path).
//!
//! Feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`,
//! composed by `mlrs_backend::prims::gram`.
//!
//! ## Why this exists (the "GEMM sums" pathology, see `kmeans.rs`)
//! `LinearRegression`'s Gram+eig path forms `G = XᵀX` (`d×d`) and `c = Xᵀy`
//! (`d×1`) — a SKINNY output over a HUGE `n_samples` reduction. Routed through
//! the generic tiled `gemm` prim (`cubek-matmul`, no split-K), this shape
//! starves the GPU of independent output tiles: `d×d` (e.g. 16×16..64×64) is
//! nowhere near enough parallel work to fill a modern GPU, no matter how large
//! `n_samples` is. This is the EXACT shape category that made KMeans'
//! `onehotᵀX` GEMM-sums "catastrophic" (see `kmeans.rs` module docs) — fixed
//! there by [`crate::kmeans::centroid_sumcount_shared`]'s row-blocked
//! shared-memory accumulation. [`gram_xty_shared`] below is the same fix
//! applied to `XᵀX`/`Xᵀy`: split `n_samples` into row BLOCKS (exposing
//! `nblocks`-way parallelism instead of `d×d`-way), have each block's cube
//! accumulate a PRIVATE partial `d×d` Gram + `d` Xty into `SharedMemory`
//! (`d <= GRAM_EIG_MAX_FEATURES = 64` in the caller, so `d² <= 4096` fits the
//! same SharedMemory budget as `jacobi_eig`/`jacobi_svd`/`kmeans`'s shared
//! kernels), then fold the (small, capped) per-block partials with
//! [`gram_xty_reduce_partials`].
//!
//! ## cubecl-cpu MLIR safety
//! Like `kmeans.rs`'s `centroid_sumcount_shared`, [`gram_xty_shared`] uses
//! `SharedMemory` — the cpu backend's MLIR lowering rejects that combined with
//! a mutable `bool` — this kernel only ever touches `F`/`u32` accumulators
//! with ascending `while` scans, but the HOST caller (`prims::gram`) still
//! gates cpu off entirely (mirrors `use_shared_sums`'s `#[cfg(feature =
//! "cpu")]` precedent) and falls back to the existing `gemm`-based formation,
//! which is already validated on cpu.

use cubecl::prelude::*;

/// Row-blocked shared-memory partial Gram (`d×d`) + Xty (`d`) accumulation —
/// stage 1. One 64-thread cube per row-block (`b < nblocks`, the RF/KMeans
/// shared-histogram idiom; the slack-cube guard is cube-uniform so barriers
/// inside are safe).
///
/// The cube keeps its PRIVATE `d × d` Gram accumulator and length-`d` Xty
/// accumulator in `SharedMemory` (fixed 4096/64-slot budget, matching the
/// `d <= 64` caller cap). For each row `i` in the block, the row is FIRST
/// staged into a length-`d` `SharedMemory` tile (`shm_row`, one cooperative
/// load: thread `t < d` loads column `t`) — every thread's `d²` Gram products
/// for that row then read `shm_row` instead of re-fetching `x[i, ·]` from
/// global once per (a, b) pair. Without this tile, `d` DIFFERENT threads each
/// re-read every one of the row's `d` elements from global memory (a
/// redundancy factor of `d`), which measurably erases the row-blocking win as
/// `d` grows (flat fit time at `d=64` vs a 34–57% win at `d=16` — the T4 A/B
/// that motivated this tile). Thread `t` OWNS Gram slots `s ≡ t (mod 64)`
/// (`s = a·d + b`, `shm_gram[s] += shm_row[a]·shm_row[b]`) — a single writer
/// per slot, so NO atomics and a DETERMINISTIC ascending-row accumulation
/// order (bitwise-reproducible). Threads `t < d` additionally own Xty slot
/// `t` (`shm_xty[t] += shm_row[t]·y[i]`). Both partials are flushed to the
/// row-block's slot of `pgram`/`pxty` at the end.
#[cube(launch)]
pub fn gram_xty_shared<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    pgram: &mut Array<F>,
    pxty: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let mut shm_gram = SharedMemory::<F>::new(4096usize);
    let mut shm_xty = SharedMemory::<F>::new(64usize);
    let mut shm_row = SharedMemory::<F>::new(64usize);
    // Linearized cube id over the (possibly Y-folded) grid — UNIFORM per
    // cube, so the slack guard below is a safe barrier scope (the RF/KMeans
    // shared-histogram idiom).
    let b = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;
    let t = UNIT_POS as u32;
    if b < nblocks {
        let dd = d * d;
        // Zero the used Gram slots (strided over the 64 threads).
        let mut s = t;
        while s < dd {
            shm_gram[s as usize] = F::new(0.0_f32);
            s += 64u32;
        }
        // Zero this thread's Xty slot (only threads t < d own one).
        if t < d {
            shm_xty[t as usize] = F::new(0.0_f32);
        }
        sync_cube();

        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut i = start;
        while i < end {
            let xbase = i * d;
            // Cooperative row tile: thread t < d loads column t of row i into
            // shared memory ONCE, so the d² Gram products below (and the Xty
            // product) read shared memory instead of re-fetching x[i, ·] from
            // global per (a, b) pair (see the function docs).
            if t < d {
                shm_row[t as usize] = x[(xbase + t) as usize];
            }
            sync_cube();

            // Gram: thread t owns slots s ≡ t (mod 64), s = a·d + bb.
            let mut s2 = t;
            while s2 < dd {
                let a = s2 / d;
                let bb = s2 % d;
                shm_gram[s2 as usize] += shm_row[a as usize] * shm_row[bb as usize];
                s2 += 64u32;
            }
            // Xty: thread t < d owns column t.
            if t < d {
                shm_xty[t as usize] += shm_row[t as usize] * y[i as usize];
            }
            // Barrier before the next row overwrites shm_row (all reads of
            // THIS row's tile above must complete first).
            sync_cube();
            i += 1u32;
        }

        // Flush the block's d × d Gram partial + length-d Xty partial to global.
        let base = b * dd;
        let mut s3 = t;
        while s3 < dd {
            pgram[(base + s3) as usize] = shm_gram[s3 as usize];
            s3 += 64u32;
        }
        if t < d {
            pxty[(b * d + t) as usize] = shm_xty[t as usize];
        }
    }
}

/// Fold the row-blocked partials of [`gram_xty_shared`] — stage 2.
///
/// One unit per Gram output element `(a, bb)` (`tid < d·d`): sum the
/// `nblocks` partial Gram sums into `gram[tid]`; units `tid < d` additionally
/// fold the Xty partials into `xty[tid]`. Ascending scans over the (small,
/// capped) `nblocks` axis only.
#[cube(launch)]
pub fn gram_xty_reduce_partials<F: Float + CubeElement>(
    pgram: &Array<F>,
    pxty: &Array<F>,
    gram: &mut Array<F>,
    xty: &mut Array<F>,
    d: u32,
    nblocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let dd = d * d;
    if tid < dd as usize {
        let mut acc = F::new(0.0_f32);
        let mut bl = 0u32;
        while bl < nblocks {
            acc += pgram[(bl * dd + tid as u32) as usize];
            bl += 1u32;
        }
        gram[tid] = acc;

        if (tid as u32) < d {
            let mut acc2 = F::new(0.0_f32);
            let mut bl2 = 0u32;
            while bl2 < nblocks {
                acc2 += pxty[(bl2 * d + tid as u32) as usize];
                bl2 += 1u32;
            }
            xty[tid] = acc2;
        }
    }
}
