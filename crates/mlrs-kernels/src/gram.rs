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

/// Gram slots one unit accumulates in registers per pass of
/// [`gram_xty_blocked`].
///
/// Eight consecutive `b` columns of ONE Gram row share the same `x[i, a]`
/// factor, so a pass loads 9 elements per row and issues 8 multiply-adds
/// (≈1.1 loads/MAC) where a slot-at-a-time sweep loads 2 per MAC. That ratio IS
/// the kernel's speed, because this reduction is global-memory-bound, not
/// FLOP-bound: the same row block is re-streamed once per slot group, so
/// grouping eight slots cuts the number of passes over it — and therefore the
/// traffic — by eight. Measured on the local wgpu adapter at
/// `n=100 000, d=64`: 40 ms → 4.8 ms.
pub const GRAM_REG_TILE: u32 = 8;

/// Row-blocked partial Gram (`d×d`) + Xty (`d`) accumulation with a
/// REGISTER-resident accumulator — one cube per row block, no shared memory
/// and no barriers.
///
/// Unit `u` walks the `d · ceil(d/8)` slot GROUPS with stride `CUBE_DIM_X`,
/// each group being up to [`GRAM_REG_TILE`] consecutive `b` columns of a single
/// Gram row `a`. For every group it holds the partial sums in REGISTERS across
/// the whole row block and stores them ONCE, so the `d²`-way output never
/// touches shared memory:
///
/// ```text
/// acc_k = Σ_{i ∈ block} x[i, a] · x[i, b₀+k]      k < min(8, d − b₀)
/// ```
///
/// ## Why this replaces [`gram_xty_shared`]
/// The shared-memory kernel stages ONE row at a time and accumulates straight
/// into a shared `d×d`, which costs **two `sync_cube` barriers per row** —
/// 512 barriers per cube at its 256-row block — and reads two shared values per
/// multiply-add. Here the row loop is the INNERMOST loop of a register
/// reduction: nothing is shared, so nothing has to be synchronized, and the
/// only traffic is the coalesced re-read of the row block itself.
///
/// The accumulator living in registers rather than a fixed `SharedMemory`
/// budget also removes the `d² ≤ 4096` cap that sent `d > 64` to the
/// starved-GEMM fallback (`prims::gram` module docs); the only remaining bound
/// is the caller's `nblocks · d²` partial buffer.
///
/// `x` is the `n × d` row-major design, `y` the length-`n` target; `pgram` and
/// `pxty` are the per-row-block partials that
/// [`gram_xty_reduce_partials`] folds. `groups_per_row = ceil(d / 8)` is passed
/// in rather than recomputed so the division is not repeated per unit.
///
/// ## Centering is fused, not a separate pass
/// `xmean` (length `d`) and `ymean` (length 1) are subtracted AS THE OPERANDS
/// ARE READ, so the centered design is never materialized: the caller neither
/// allocates the `n × d` centered copy nor pays the write + re-read that
/// `center_columns` costs around it. Both means are hoisted out of the row
/// loop, so the fusion adds `d + 1` loads per group rather than per row, and
/// the subtraction happens on the DATA (never as an `XᵀX − n·x̄x̄ᵀ` correction,
/// which loses catastrophic precision in `f32` on a design whose column means
/// dominate its spread). Pass an all-zero `xmean`/`ymean` for the raw Gram.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gram_xty_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xmean: &Array<F>,
    ymean: &Array<F>,
    pgram: &mut Array<F>,
    pxty: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
    groups_per_row: u32,
) {
    // Linearized cube id over the (possibly Y-folded) grid — UNIFORM per cube.
    let b = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;
    if b < nblocks {
        let t = UNIT_POS as u32;
        let stride = CUBE_DIM_X;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let ng = d * groups_per_row;
        let gbase = b * d * d;

        let mut g = t;
        while g < ng {
            let a = g / groups_per_row;
            let b0 = (g % groups_per_row) * 8u32;
            let mut cnt = d - b0;
            if cnt > 8u32 {
                cnt = 8u32;
            }
            let sbase = gbase + a * d + b0;

            // Both operands' means are loop-invariant, so they are loaded once
            // per GROUP rather than once per row.
            let ma = xmean[a as usize];

            // SYMMETRY: `XᵀX` is symmetric, so only the groups that reach the
            // lower triangle are accumulated — half the multiply-adds, and
            // `gram_xty_reduce_partials` reads the mirrored slot for the upper
            // half. The whole cube walks `g` in lockstep, so a skipped range is
            // skipped by every unit at once and the elision costs no load
            // balance. A kept group may straddle the diagonal; its handful of
            // above-diagonal lanes are simply computed correctly and stored,
            // which is harmless.
            if b0 > a {
                // skipped
            } else if cnt == 8u32 {
                // Full tile: eight independent register chains, one shared
                // `x[i, a]` load. This is the path every `d % 8 == 0` shape
                // (16 / 32 / 64 / 256 …) takes for ALL of its groups.
                let m0 = xmean[b0 as usize];
                let m1 = xmean[(b0 + 1u32) as usize];
                let m2 = xmean[(b0 + 2u32) as usize];
                let m3 = xmean[(b0 + 3u32) as usize];
                let m4 = xmean[(b0 + 4u32) as usize];
                let m5 = xmean[(b0 + 5u32) as usize];
                let m6 = xmean[(b0 + 6u32) as usize];
                let m7 = xmean[(b0 + 7u32) as usize];
                let mut a0 = F::new(0.0_f32);
                let mut a1 = F::new(0.0_f32);
                let mut a2 = F::new(0.0_f32);
                let mut a3 = F::new(0.0_f32);
                let mut a4 = F::new(0.0_f32);
                let mut a5 = F::new(0.0_f32);
                let mut a6 = F::new(0.0_f32);
                let mut a7 = F::new(0.0_f32);
                let mut i = start;
                while i < end {
                    let xb = i * d + b0;
                    let xa = x[(i * d + a) as usize] - ma;
                    a0 += xa * (x[xb as usize] - m0);
                    a1 += xa * (x[(xb + 1u32) as usize] - m1);
                    a2 += xa * (x[(xb + 2u32) as usize] - m2);
                    a3 += xa * (x[(xb + 3u32) as usize] - m3);
                    a4 += xa * (x[(xb + 4u32) as usize] - m4);
                    a5 += xa * (x[(xb + 5u32) as usize] - m5);
                    a6 += xa * (x[(xb + 6u32) as usize] - m6);
                    a7 += xa * (x[(xb + 7u32) as usize] - m7);
                    i += 1u32;
                }
                pgram[sbase as usize] = a0;
                pgram[(sbase + 1u32) as usize] = a1;
                pgram[(sbase + 2u32) as usize] = a2;
                pgram[(sbase + 3u32) as usize] = a3;
                pgram[(sbase + 4u32) as usize] = a4;
                pgram[(sbase + 5u32) as usize] = a5;
                pgram[(sbase + 6u32) as usize] = a6;
                pgram[(sbase + 7u32) as usize] = a7;
            } else {
                // Ragged tail (`d % 8 != 0`): at most one group per Gram row,
                // walked a slot at a time so the store count stays exact.
                let mut k = 0u32;
                while k < cnt {
                    let mk = xmean[(b0 + k) as usize];
                    let mut acc = F::new(0.0_f32);
                    let mut i = start;
                    while i < end {
                        let xb = i * d;
                        acc += (x[(xb + a) as usize] - ma) * (x[(xb + b0 + k) as usize] - mk);
                        i += 1u32;
                    }
                    pgram[(sbase + k) as usize] = acc;
                    k += 1u32;
                }
            }
            g += stride;
        }

        // Xty: unit `u` owns columns `c ≡ u (mod CUBE_DIM_X)`, same register
        // reduction over the block's rows.
        let my = ymean[0];
        let mut c = t;
        while c < d {
            let mc = xmean[c as usize];
            let mut acc = F::new(0.0_f32);
            let mut i = start;
            while i < end {
                acc += (x[(i * d + c) as usize] - mc) * (y[i as usize] - my);
                i += 1u32;
            }
            pxty[(b * d + c) as usize] = acc;
            c += stride;
        }
    }
}

/// Output tile edge one unit of [`gram_xty_tiled`] owns, in each of the two
/// Gram axes.
///
/// A `4 × 4` tile turns 8 loads per row into 16 multiply-adds. That ratio is
/// NOT why it is fast, though — [`gram_xty_blocked`]'s `1 × 8` tile already
/// reaches 1.125 loads/MAC, and both kernels are bound by something else
/// entirely: the number of times the cube re-streams its row block from
/// memory, which is `ceil(tiles / CUBE_DIM)`. Squaring the tile squares the
/// slot coverage per pass, so at `d = 256` the lower-triangle tile count drops
/// from `d · d/8 = 8192` to `T(T+1)/2 = 2080` with `T = d/4` — 4× fewer passes
/// over the design, on top of the 2× the wider cube buys.
pub const GRAM_TILE: u32 = 4;

/// Row-blocked partial Gram (`d×d`) + Xty (`d`) accumulation with a
/// TWO-DIMENSIONAL register tile — one cube per row block, no shared memory
/// and no barriers.
///
/// ## What changes against [`gram_xty_blocked`]
/// That kernel gives each unit ONE Gram row and eight consecutive columns of
/// it, so a cube of 64 units covers 64 of the `d · ceil(d/8)` slot groups per
/// pass and has to re-read the whole row block `ceil(d · ceil(d/8) / 64)`
/// times — 128 times at `d = 256`. Since the reduction is memory-bound, that
/// re-streaming IS the cost: 128 passes over a 102 MiB design is 13 GiB of
/// traffic to produce a 256 KiB Gram.
///
/// Here a unit owns a [`GRAM_TILE`]`²` SQUARE of the Gram instead, and the
/// units are handed only the tiles that intersect the lower triangle
/// (`ti ≥ tj`), so a `d = 256` fit walks 2080 tiles rather than 8192 groups.
/// With a 256-unit cube that is 8 passes instead of 128.
///
/// The tile index is decoded to `(ti, tj)` by advancing a running triangular
/// number rather than by `sqrt` — `t` only ever increases within a unit, so the
/// walk is `O(T + tiles/CUBE_DIM)` integer adds in total, and it stays exact
/// (and stays off the f64-transcendental paths some adapters lack).
///
/// `xmean` / `ymean` are subtracted as the operands are read, exactly as in
/// [`gram_xty_blocked`]; pass all-zero means for the raw Gram. `Xᵀy` is
/// accumulated by the DIAGONAL tiles only (tile `(ti, ti)` owns columns
/// `4·ti … 4·ti+3`, which partitions the columns exactly once), so it costs
/// `T` extra row-block reads out of `T(T+1)/2` rather than a whole extra pass.
///
/// Only the lower triangle is written; `gram_xty_reduce_partials` with
/// `lower_only = 1` mirrors the rest.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gram_xty_tiled<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xmean: &Array<F>,
    ymean: &Array<F>,
    pgram: &mut Array<F>,
    pxty: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
    ntiles: u32,
) {
    let bidx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;
    if bidx < nblocks {
        let stride = CUBE_DIM_X;
        let start = bidx * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let gbase = bidx * d * d;
        let my = ymean[0];

        // Running triangular decode: `ti` is the tile row and `tri` is
        // `ti·(ti+1)/2`, the index of that row's first tile. `t` is
        // non-decreasing within this unit, so the inner advance is amortized.
        let mut ti = 0u32;
        let mut tri = 0u32;

        let mut t = UNIT_POS as u32;
        while t < ntiles {
            while tri + ti + 1u32 <= t {
                tri += ti + 1u32;
                ti += 1u32;
            }
            let tj = t - tri;
            let a0 = ti * 4u32;
            let b0 = tj * 4u32;
            let mut ca = d - a0;
            if ca > 4u32 {
                ca = 4u32;
            }
            let mut cb = d - b0;
            if cb > 4u32 {
                cb = 4u32;
            }

            if ca == 4u32 && cb == 4u32 {
                let ma0 = xmean[a0 as usize];
                let ma1 = xmean[(a0 + 1u32) as usize];
                let ma2 = xmean[(a0 + 2u32) as usize];
                let ma3 = xmean[(a0 + 3u32) as usize];
                let mb0 = xmean[b0 as usize];
                let mb1 = xmean[(b0 + 1u32) as usize];
                let mb2 = xmean[(b0 + 2u32) as usize];
                let mb3 = xmean[(b0 + 3u32) as usize];

                let mut c00 = F::new(0.0_f32);
                let mut c01 = F::new(0.0_f32);
                let mut c02 = F::new(0.0_f32);
                let mut c03 = F::new(0.0_f32);
                let mut c10 = F::new(0.0_f32);
                let mut c11 = F::new(0.0_f32);
                let mut c12 = F::new(0.0_f32);
                let mut c13 = F::new(0.0_f32);
                let mut c20 = F::new(0.0_f32);
                let mut c21 = F::new(0.0_f32);
                let mut c22 = F::new(0.0_f32);
                let mut c23 = F::new(0.0_f32);
                let mut c30 = F::new(0.0_f32);
                let mut c31 = F::new(0.0_f32);
                let mut c32 = F::new(0.0_f32);
                let mut c33 = F::new(0.0_f32);

                let mut i = start;
                while i < end {
                    let base = i * d;
                    let xa0 = x[(base + a0) as usize] - ma0;
                    let xa1 = x[(base + a0 + 1u32) as usize] - ma1;
                    let xa2 = x[(base + a0 + 2u32) as usize] - ma2;
                    let xa3 = x[(base + a0 + 3u32) as usize] - ma3;
                    let xb0 = x[(base + b0) as usize] - mb0;
                    let xb1 = x[(base + b0 + 1u32) as usize] - mb1;
                    let xb2 = x[(base + b0 + 2u32) as usize] - mb2;
                    let xb3 = x[(base + b0 + 3u32) as usize] - mb3;
                    c00 += xa0 * xb0;
                    c01 += xa0 * xb1;
                    c02 += xa0 * xb2;
                    c03 += xa0 * xb3;
                    c10 += xa1 * xb0;
                    c11 += xa1 * xb1;
                    c12 += xa1 * xb2;
                    c13 += xa1 * xb3;
                    c20 += xa2 * xb0;
                    c21 += xa2 * xb1;
                    c22 += xa2 * xb2;
                    c23 += xa2 * xb3;
                    c30 += xa3 * xb0;
                    c31 += xa3 * xb1;
                    c32 += xa3 * xb2;
                    c33 += xa3 * xb3;
                    i += 1u32;
                }

                let r0 = gbase + a0 * d + b0;
                pgram[r0 as usize] = c00;
                pgram[(r0 + 1u32) as usize] = c01;
                pgram[(r0 + 2u32) as usize] = c02;
                pgram[(r0 + 3u32) as usize] = c03;
                let r1 = r0 + d;
                pgram[r1 as usize] = c10;
                pgram[(r1 + 1u32) as usize] = c11;
                pgram[(r1 + 2u32) as usize] = c12;
                pgram[(r1 + 3u32) as usize] = c13;
                let r2 = r1 + d;
                pgram[r2 as usize] = c20;
                pgram[(r2 + 1u32) as usize] = c21;
                pgram[(r2 + 2u32) as usize] = c22;
                pgram[(r2 + 3u32) as usize] = c23;
                let r3 = r2 + d;
                pgram[r3 as usize] = c30;
                pgram[(r3 + 1u32) as usize] = c31;
                pgram[(r3 + 2u32) as usize] = c32;
                pgram[(r3 + 3u32) as usize] = c33;
            } else {
                // Ragged edge tile (`d % 4 != 0`): at most one tile row and one
                // tile column of the grid, walked a slot at a time.
                let mut k = 0u32;
                while k < ca {
                    let mk = xmean[(a0 + k) as usize];
                    let mut l = 0u32;
                    while l < cb {
                        let ml = xmean[(b0 + l) as usize];
                        let mut acc = F::new(0.0_f32);
                        let mut i = start;
                        while i < end {
                            let base = i * d;
                            acc += (x[(base + a0 + k) as usize] - mk)
                                * (x[(base + b0 + l) as usize] - ml);
                            i += 1u32;
                        }
                        pgram[(gbase + (a0 + k) * d + b0 + l) as usize] = acc;
                        l += 1u32;
                    }
                    k += 1u32;
                }
            }

            // `Xᵀy` rides on the diagonal tiles: tile `(ti, ti)` owns exactly
            // columns `a0 … a0+ca-1`, and the diagonal tiles partition the
            // columns, so every column is accumulated once and only once.
            if ti == tj {
                let mut k = 0u32;
                while k < ca {
                    let mk = xmean[(a0 + k) as usize];
                    let mut acc = F::new(0.0_f32);
                    let mut i = start;
                    while i < end {
                        acc += (x[(i * d + a0 + k) as usize] - mk) * (y[i as usize] - my);
                        i += 1u32;
                    }
                    pxty[(bidx * d + a0 + k) as usize] = acc;
                    k += 1u32;
                }
            }

            t += stride;
        }
    }
}

/// Row-blocked partial column sums of `x` (`d`) and of `y` (1) — the mean pass
/// [`gram_xty_blocked`]'s fused centering needs, in the SAME cube-per-row-block
/// shape.
///
/// Unit `u` owns columns `c ≡ u (mod CUBE_DIM_X)` and folds them over the
/// block's rows in a register, exactly as the Xty tail of
/// [`gram_xty_blocked`] does. Adjacent units therefore read ADJACENT addresses
/// of the same row — which is the whole point: `prims::center`'s column-mean
/// walks one column at a time with a `d`-element stride, so on a row-major
/// design every read is its own cache line and the pass gets slower as `d`
/// grows (measured 151 ms at `n = 100 000, d = 256`, against 9 ms at `d = 64`
/// for a quarter of the bytes).
///
/// Unit 0 additionally folds `y` for the target mean.
#[cube(launch)]
pub fn col_sums_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    psum: &mut Array<F>,
    pysum: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let b = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;
    if b < nblocks {
        let t = UNIT_POS as u32;
        let stride = CUBE_DIM_X;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }

        let mut c = t;
        while c < d {
            let mut acc = F::new(0.0_f32);
            let mut i = start;
            while i < end {
                acc += x[(i * d + c) as usize];
                i += 1u32;
            }
            psum[(b * d + c) as usize] = acc;
            c += stride;
        }

        if t == 0u32 {
            let mut accy = F::new(0.0_f32);
            let mut i = start;
            while i < end {
                accy += y[i as usize];
                i += 1u32;
            }
            pysum[b as usize] = accy;
        }
    }
}

/// Fold [`col_sums_blocked`]'s partials into the column MEANS — one unit per
/// output column (`tid < d`), plus unit 0 for the target mean.
///
/// `inv_n` is `1/n` supplied by the host so the division happens once per
/// output rather than per partial.
#[cube(launch)]
pub fn col_sums_reduce<F: Float + CubeElement>(
    psum: &Array<F>,
    pysum: &Array<F>,
    xmean: &mut Array<F>,
    ymean: &mut Array<F>,
    d: u32,
    nblocks: u32,
    inv_n: F,
) {
    let tid = ABSOLUTE_POS;
    if tid < d as usize {
        let mut acc = F::new(0.0_f32);
        let mut bl = 0u32;
        while bl < nblocks {
            acc += psum[(bl * d + tid as u32) as usize];
            bl += 1u32;
        }
        xmean[tid] = acc * inv_n;

        if tid == 0usize {
            let mut accy = F::new(0.0_f32);
            let mut bl2 = 0u32;
            while bl2 < nblocks {
                accy += pysum[bl2 as usize];
                bl2 += 1u32;
            }
            ymean[0] = accy * inv_n;
        }
    }
}

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
///
/// `lower_only = 1` means the stage-1 kernel filled only the lower triangle of
/// each partial ([`gram_xty_blocked`]'s symmetry elision): an above-diagonal
/// output then folds its MIRROR slot `(bb, a)` instead, which is the same sum
/// by symmetry of `XᵀX`. `lower_only = 0` reads `tid` directly, for the
/// full-triangle [`gram_xty_shared`] partials.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gram_xty_reduce_partials<F: Float + CubeElement>(
    pgram: &Array<F>,
    pxty: &Array<F>,
    gram: &mut Array<F>,
    xty: &mut Array<F>,
    d: u32,
    nblocks: u32,
    lower_only: u32,
) {
    let tid = ABSOLUTE_POS;
    let dd = d * d;
    if tid < dd as usize {
        let a = (tid as u32) / d;
        let bb = (tid as u32) % d;
        let mut src = tid as u32;
        if lower_only == 1u32 && bb > a {
            src = bb * d + a;
        }
        let mut acc = F::new(0.0_f32);
        let mut bl = 0u32;
        while bl < nblocks {
            acc += pgram[(bl * dd + src) as usize];
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
