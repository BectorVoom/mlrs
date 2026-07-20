//! `kmeans` — Lloyd centroid-update + inertia `#[cube]` kernels (CLUSTER-01,
//! D-01).
//!
//! Feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`
//! for the n-heavy parts of the Lloyd iteration, composed by
//! `mlrs_backend::prims::kmeans`. The original two ([`centroid_sumcount`],
//! [`inertia_rows`]) are kept for their prim tests; the DEVICE-RESIDENT Lloyd
//! hot loop uses the fused/blocked set below ([`dist_direct_2d`] +
//! [`argmin_dist_rows`],
//! [`centroid_sumcount_blocked`] + [`centroid_reduce_partials`],
//! [`labels_diff_blocked`], [`block_sum_f`], [`gather_rows_idx`],
//! [`kmeanspp_mind2`], [`col_sum_blocked`] + [`col_sqdiff_blocked`]) so the
//! per-iteration host traffic is a few KB of sums/counts, never `O(n)`:
//!
//! - [`centroid_sumcount`] — per (centroid `c`, feature `j`) GATHER: scan all
//!   `n` rows and accumulate `Σ X[i, j]` over rows whose `label[i] == c` plus
//!   the per-centroid count. One unit per `(c, j)` output element reads every
//!   row (a GATHER, never a scatter) so there is NO atomic / no race — the host
//!   then divides each sum by its count to get the mean. This is the n-heavy
//!   accumulation (the `center_columns` `c = tid % cols` per-element map
//!   generalised to a per-output gather).
//! - [`inertia_rows`] — per ROW `i` GATHER: `Σ_j (X[i, j] − centers[label_i,
//!   j])²`, the squared distance from each sample to its ASSIGNED center (the
//!   `reduce_sumsq` squared-norm accumulation specialised to the assigned
//!   center). The host then sums the `n` per-row partials to the scalar inertia
//!   `Σ_i ‖X_i − centers[labels_i]‖²`.
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! cpu(f64) is the primary gate and its MLIR `run pass` lowering REJECTS
//! `SharedMemory` combined with mutable `bool` flags, the `F::INFINITY` const,
//! and descending-shift loops (proven in plan 05-02). Both kernels here use ONLY
//! `F`/`u32` accumulators and ascending `while` scans with `if` guards — no
//! `SharedMemory`, no mutable `bool`, no infinity sentinel, no descending shift
//! — so they LAUNCH (not just compile) on cubecl-cpu. The per-output GATHER
//! layout also avoids cross-unit atomics, which the cpu backend does not lower
//! reliably.
//!
//! All kernels carry NO backend feature (D-13). Empty-cluster relocation and the
//! final mean/sum finalize are HOST concerns (small-k reductions, RESEARCH Open
//! Q1) handled in `prims::kmeans`. Tests live in
//! `crates/mlrs-backend/tests/{kmeanspp,lloyd}_test.rs` (AGENTS.md §2).

use cubecl::prelude::*;

pub use self::centroid_sumcount as kmeans_centroid_sumcount;
pub use self::inertia_rows as kmeans_inertia_rows;

/// Per-centroid feature SUM + per-centroid COUNT for the Lloyd centroid update
/// (CLUSTER-01).
///
/// One unit per `(c, j)` output element (`c` in `0..k`, `j` in `0..d`,
/// `ABSOLUTE_POS = c * d + j`): scan ALL `n` rows of the row-major `x`
/// (`n × d`) and accumulate `Σ X[i, j]` over the rows whose `labels[i] == c`.
/// Unit `(c, 0)` additionally writes the per-centroid count into `counts[c]`.
///
/// - `x` is the row-major `n × d` sample matrix.
/// - `labels` is the length-`n` assignment (`u32`, one cluster id per row).
/// - `sums` is the `k × d` output (per-centroid feature sums).
/// - `counts` is the length-`k` output (per-centroid assigned-row count).
/// - `n`, `d`, `k` are scalar `u32` args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg`, like `dist_combine_clamp`'s `rows`/`cols`).
///
/// This is a GATHER (each output reads many inputs), so there is no scatter
/// race and no atomic — cubecl-cpu lowers it directly. The host divides
/// `sums[c, j]` by `counts[c]` to form the centroid mean (and relocates empty
/// clusters first, so `counts[c] == 0` never reaches a divide).
#[cube(launch)]
pub fn centroid_sumcount<F: Float + CubeElement>(
    x: &Array<F>,
    labels: &Array<u32>,
    sums: &mut Array<F>,
    counts: &mut Array<u32>,
    n: u32,
    d: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * d;
    if tid < total as usize {
        let c = (tid as u32) / d;
        let j = (tid as u32) % d;

        // GATHER: accumulate Σ X[i, j] over rows with label == c. Only F/u32
        // accumulators + an ascending scan + an `if` guard — no SharedMemory, no
        // mutable bool, no atomic (cubecl-cpu MLIR safe, plan 05-02).
        let mut acc = F::new(0.0_f32);
        let mut cnt = 0u32;
        let mut i = 0u32;
        while i < n {
            if labels[i as usize] == c {
                acc += x[(i * d + j) as usize];
                cnt += 1u32;
            }
            i += 1u32;
        }
        sums[tid] = acc;
        // The first feature unit of each centroid records the count once.
        if j == 0u32 {
            counts[c as usize] = cnt;
        }
    }
}

/// Per-row squared distance to the ASSIGNED center, for the inertia accumulation
/// (CLUSTER-01).
///
/// One unit per ROW `i` (`ABSOLUTE_POS = i`): compute `Σ_j (X[i, j] −
/// centers[labels_i, j])²` — the squared Euclidean distance from sample `i` to
/// its assigned center. The host then sums the length-`n` `out` partials to the
/// scalar inertia `Σ_i ‖X_i − centers[labels_i]‖²` (no sqrt — squared, Pitfall
/// 8 / D-08).
///
/// - `x` is the row-major `n × d` sample matrix.
/// - `centers` is the row-major `k × d` centroid matrix.
/// - `labels` is the length-`n` assignment (`u32`).
/// - `out` is the length-`n` per-row squared distance.
/// - `n`, `d` are scalar `u32` args passed BY VALUE.
///
/// A GATHER over the assigned center's `d` features (the `reduce_sumsq`
/// squared-difference accumulation), only F/u32 accumulators + an ascending
/// scan — cubecl-cpu MLIR safe.
#[cube(launch)]
pub fn inertia_rows<F: Float + CubeElement>(
    x: &Array<F>,
    centers: &Array<F>,
    labels: &Array<u32>,
    out: &mut Array<F>,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let c = labels[i];
        let xbase = (i as u32) * d;
        let cbase = c * d;
        let mut acc = F::new(0.0_f32);
        let mut j = 0u32;
        while j < d {
            let diff = x[(xbase + j) as usize] - centers[(cbase + j) as usize];
            acc += diff * diff;
            j += 1u32;
        }
        out[i] = acc;
    }
}

/// FUSED nearest-center assignment: per ROW `i`, the direct
/// `argmin_c Σ_j (X[i,j] − centers[c,j])²` plus the winning squared distance.
///
/// One unit per row: loop the `k` centers, accumulate the direct squared
/// Euclidean distance over the `d` features, and keep the smallest with a
/// STRICT `<` compare (lowest-index tie-break, D-02). Writes `labels[i]`
/// (`u32` in `0..k`) and `dist[i]` (the min squared distance — exactly the
/// per-row inertia term, so the Lloyd loop's empty-cluster relocation and the
/// final inertia reuse this buffer with NO extra pass).
///
/// This (with [`argmin_dist_rows`]) replaces the launch-per-row `argmin_rows`
/// + GEMM-expansion `distance` assignment path in the KMeans hot loop: no
/// `row_reduce(Shared)` norm term runs (the PyO3 landmine) and the labels stay
/// DEVICE-resident. The direct form is also numerically tighter than the GEMM
/// expansion (no `‖x‖² + ‖c‖² − 2x·c` cancellation, so no clamp is needed).
///
/// The assignment is deliberately SPLIT into two short-single-loop kernels —
/// this per-`(i, c)` distance (a `d`-iteration loop) into an `n × k` staging
/// matrix, then the per-row `k`-iteration argmin — instead of one fused
/// per-row `k × d` nested loop: the fused nested-loop form compiled
/// PATHOLOGICALLY under wgpu/naga (~12× slower per visit than the identical
/// work in single-loop kernels, measured on the perf ladder), the same
/// "fine-grained kernels + staging buffers beat mega-fusion" finding as the RF
/// best-split kernel. Consecutive units share the row `i` (adjacent `c`), so
/// the `x` row loads broadcast and the small `centers` matrix stays cached.
#[cube(launch)]
pub fn dist_direct_2d<F: Float + CubeElement>(
    x: &Array<F>,
    centers: &Array<F>,
    dmat: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let xbase = i * d;
        let cbase = c * d;
        let mut acc = F::new(0.0_f32);
        let mut j = 0u32;
        while j < d {
            let diff = x[(xbase + j) as usize] - centers[(cbase + j) as usize];
            acc += diff * diff;
            j += 1u32;
        }
        dmat[tid] = acc;
    }
}

/// 4-center-chunked variant of [`dist_direct_2d`]: one unit per `(row i,
/// center-chunk cq)` accumulates the squared distances to centers `4cq ..
/// 4cq+4` in four register accumulators, loading each `x[i, j]` ONCE per
/// chunk — a 4× cut in the dominant `x` traffic (the plain per-`(i, c)` form
/// re-reads the row for every center). Trailing-chunk lanes are guarded per
/// center. Single ascending `d`-loop (short-loop shape; no nested `k × d`).
#[cube(launch)]
pub fn dist_direct_2d_c4<F: Float + CubeElement>(
    x: &Array<F>,
    centers: &Array<F>,
    dmat: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let kq = (k + 3u32) / 4u32;
    let total = n * kq;
    if tid < total as usize {
        let i = (tid as u32) / kq;
        let cq = (tid as u32) % kq;
        let c0 = cq * 4u32;
        let xbase = i * d;
        let cb = c0 * d;
        let mut a0 = F::new(0.0_f32);
        let mut a1 = F::new(0.0_f32);
        let mut a2 = F::new(0.0_f32);
        let mut a3 = F::new(0.0_f32);
        // Tail guard hoisted OUT of the j-loop (uniform per thread): full
        // chunks — the common case — run the unguarded 4-wide loop; only the
        // final ragged chunk pays per-center guards. In-loop guards compiled
        // pathologically on wgpu (~20× slower assign).
        if c0 + 4u32 <= k {
            let mut j = 0u32;
            while j < d {
                let xv = x[(xbase + j) as usize];
                let d0 = xv - centers[(cb + j) as usize];
                a0 += d0 * d0;
                let d1 = xv - centers[(cb + d + j) as usize];
                a1 += d1 * d1;
                let d2 = xv - centers[(cb + 2u32 * d + j) as usize];
                a2 += d2 * d2;
                let d3 = xv - centers[(cb + 3u32 * d + j) as usize];
                a3 += d3 * d3;
                j += 1u32;
            }
        } else {
            let mut j = 0u32;
            while j < d {
                let xv = x[(xbase + j) as usize];
                let d0 = xv - centers[(cb + j) as usize];
                a0 += d0 * d0;
                if c0 + 1u32 < k {
                    let d1 = xv - centers[(cb + d + j) as usize];
                    a1 += d1 * d1;
                }
                if c0 + 2u32 < k {
                    let d2 = xv - centers[(cb + 2u32 * d + j) as usize];
                    a2 += d2 * d2;
                }
                j += 1u32;
            }
        }
        let base = i * k + c0;
        dmat[base as usize] = a0;
        if c0 + 1u32 < k {
            dmat[(base + 1u32) as usize] = a1;
        }
        if c0 + 2u32 < k {
            dmat[(base + 2u32) as usize] = a2;
        }
        if c0 + 3u32 < k {
            dmat[(base + 3u32) as usize] = a3;
        }
    }
}

/// Per-row argmin over the [`dist_direct_2d`] staging matrix: one unit per row
/// `i`, an ascending `k`-iteration scan with a STRICT `<` compare
/// (lowest-index tie-break, D-02). Writes `labels[i]` (`u32` in `0..k`) and
/// `dist[i]` (the winning squared distance — exactly the per-row inertia term,
/// so empty-cluster relocation and the final inertia reuse this buffer with NO
/// extra pass). `best` is seeded from center 0 inside the scan (`c == 0`
/// overwrite) — no `F::INFINITY` sentinel (cubecl-cpu MLIR safe, plan 05-02).
#[cube(launch)]
pub fn argmin_dist_rows<F: Float + CubeElement>(
    dmat: &Array<F>,
    labels: &mut Array<u32>,
    dist: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = (i as u32) * k;
        let mut best = F::new(0.0_f32);
        let mut best_c = 0u32;
        let mut c = 0u32;
        while c < k {
            let v = dmat[(base + c) as usize];
            if c == 0u32 {
                best = v;
            } else if v < best {
                best = v;
                best_c = c;
            }
            c += 1u32;
        }
        labels[i] = best_c;
        dist[i] = best;
    }
}

/// ROW-BLOCKED per-centroid feature sum + count — stage 1 of the Lloyd update.
///
/// One unit per `(block b, centroid c, feature j)` (`ABSOLUTE_POS = b·k·d +
/// c·d + j`): scan ONLY the rows of block `b` (`[b·rows_per_block, min(n,
/// (b+1)·rows_per_block))`) and accumulate `Σ X[i,j]` over rows with
/// `labels[i] == c` into `psums[b·k·d + c·d + j]`; the `j == 0` unit records
/// the block's per-centroid count into `pcounts[b·k + c]`. Stage 2
/// ([`centroid_reduce_partials`]) folds the `nblocks` partials.
///
/// This replaces the single-block [`centroid_sumcount`] in the hot loop: that
/// kernel exposes only `k·d` units of parallelism, each scanning ALL `n` rows
/// — a few hundred busy threads on a GPU with tens of thousands of lanes. The
/// blocked layout exposes `nblocks·k·d` units. Consecutive units within a
/// 256-thread cube share the block `b` (for `k·d ≥ 256`), so the per-row
/// `labels[i]` load is a broadcast and the `x[i·d + j]` loads coalesce over
/// `j`. GATHER only — no atomic, no SharedMemory (cubecl-cpu MLIR safe).
#[cube(launch)]
pub fn centroid_sumcount_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    labels: &Array<u32>,
    psums: &mut Array<F>,
    pcounts: &mut Array<u32>,
    n: u32,
    d: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * k * d;
    if tid < total as usize {
        let b = (tid as u32) / (k * d);
        let rem = (tid as u32) % (k * d);
        let c = rem / d;
        let j = rem % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut cnt = 0u32;
        let mut i = start;
        while i < end {
            if labels[i as usize] == c {
                acc += x[(i * d + j) as usize];
                cnt += 1u32;
            }
            i += 1u32;
        }
        psums[tid] = acc;
        if j == 0u32 {
            pcounts[(b * k + c) as usize] = cnt;
        }
    }
}

/// SHARED-MEMORY row-blocked per-centroid feature sums — the O(n·d) stage-1
/// alternative to the O(n·k·d) [`centroid_sumcount_blocked`] gather for the
/// GPU backends (host-gated to `k·d ≤ 4096`; the cpu backend keeps the gather
/// — its MLIR lowering rejects `SharedMemory`).
///
/// One 64-thread cube per row block. The cube keeps the whole `k × d` partial
/// accumulator in a fixed 4096-slot `SharedMemory<F>` (16 KiB for f32) and
/// each thread OWNS the feature columns `j ≡ UNIT_POS (mod 64)`: for every
/// row of the block, thread `t` adds `x[i, j]` into `shm[label_i·d + j]` for
/// its own columns only — a single writer per slot, so NO atomics and a
/// DETERMINISTIC ascending-row accumulation order (bitwise-reproducible,
/// unlike a float `atomic_add` design). The flushed `psums` partials are
/// folded by the same [`centroid_reduce_partials`] as the gather path (the
/// per-block counts come from the separate [`count_blocked`] pass on the same
/// block layout). Barriers are cube-uniform: the slack-cube guard `b <
/// nblocks` is uniform per cube (the RF shared-histogram precedent).
#[cube(launch)]
pub fn centroid_sumcount_shared<F: Float + CubeElement>(
    x: &Array<F>,
    labels: &Array<u32>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let mut shm = SharedMemory::<F>::new(4096usize);
    // Linearized cube id over the (possibly Y-folded) grid — the RF
    // shared-histogram idiom; UNIFORM per cube, so the slack guard below is a
    // safe barrier scope.
    let b = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;
    let t = UNIT_POS as u32;
    if b < nblocks {
        let kd = k * d;
        // Zero the used slots (strided over the 64 threads).
        let mut s = t;
        while s < kd {
            shm[s as usize] = F::new(0.0_f32);
            s += 64u32;
        }
        sync_cube();

        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut i = start;
        while i < end {
            let c = labels[i as usize];
            let cbase = c * d;
            let xbase = i * d;
            // Thread t owns columns j ≡ t (mod 64) — single writer per slot.
            let mut j = t;
            while j < d {
                shm[(cbase + j) as usize] += x[(xbase + j) as usize];
                j += 64u32;
            }
            i += 1u32;
        }
        sync_cube();

        // Flush the block's k × d partial to global.
        let base = b * kd;
        let mut s2 = t;
        while s2 < kd {
            psums[(base + s2) as usize] = shm[s2 as usize];
            s2 += 64u32;
        }
    }
}

/// Fold the row-blocked partials of [`centroid_sumcount_blocked`] — stage 2.
///
/// One unit per `(c, j)` output element: sum the `nblocks` partial sums into
/// `sums[c·d + j]`; the `j == 0` unit additionally folds the per-block counts
/// into `counts[c]`. Ascending scans over the (small) `nblocks` axis only.
#[cube(launch)]
pub fn centroid_reduce_partials<F: Float + CubeElement>(
    psums: &Array<F>,
    pcounts: &Array<u32>,
    sums: &mut Array<F>,
    counts: &mut Array<u32>,
    d: u32,
    k: u32,
    nblocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * d;
    if tid < total as usize {
        let c = (tid as u32) / d;
        let j = (tid as u32) % d;
        let mut acc = F::new(0.0_f32);
        let mut b = 0u32;
        while b < nblocks {
            acc += psums[(b * total + tid as u32) as usize];
            b += 1u32;
        }
        sums[tid] = acc;
        if j == 0u32 {
            let mut cnt = 0u32;
            let mut b2 = 0u32;
            while b2 < nblocks {
                cnt += pcounts[(b2 * k + c) as usize];
                b2 += 1u32;
            }
            counts[c as usize] = cnt;
        }
    }
}

/// Row-blocked count of positions where two `u32` label buffers DIFFER — the
/// device-resident strict-convergence (`array_equal`) check of the Lloyd loop.
///
/// One unit per block: scan its row range and count `a[i] != b[i]`, writing
/// the per-block count to `out[out_offset + block]`. The host sums the (tiny)
/// `nblocks` partials; `0` ⇒ the labeling did not change (sklearn's strict
/// break, Pitfall 6). `out_offset` lets the caller pack these partials behind
/// other `u32` results in one readback buffer.
#[cube(launch)]
pub fn labels_diff_blocked(
    a: &Array<u32>,
    b: &Array<u32>,
    out: &mut Array<u32>,
    n: u32,
    nblocks: u32,
    rows_per_block: u32,
    out_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nblocks as usize {
        let start = (tid as u32) * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut cnt = 0u32;
        let mut i = start;
        while i < end {
            if a[i as usize] != b[i as usize] {
                cnt += 1u32;
            }
            i += 1u32;
        }
        out[(out_offset + tid as u32) as usize] = cnt;
    }
}

/// Row-blocked partial sum of a length-`n` `F` vector (one unit per block,
/// ascending scan). The host sums the `nblocks` partials — used to fold the
/// per-row assigned-center distances ([`assign_min_rows`]'s `dist`) into the
/// scalar inertia WITHOUT an `n`-sized readback.
#[cube(launch)]
pub fn block_sum_f<F: Float + CubeElement>(
    v: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nblocks as usize {
        let start = (tid as u32) * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            acc += v[i as usize];
            i += 1u32;
        }
        out[tid] = acc;
    }
}

/// Gather `k` rows of the row-major `x` (`n × d`) by index into a `k × d`
/// output (one unit per output element) — forms the k-means++ init centers on
/// the device, so the estimator never reads the full `x` back just to copy
/// `k` rows.
#[cube(launch)]
pub fn gather_rows_idx<F: Float + CubeElement>(
    x: &Array<F>,
    idx: &Array<u32>,
    out: &mut Array<F>,
    d: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * d;
    if tid < total as usize {
        let c = (tid as u32) / d;
        let j = (tid as u32) % d;
        out[tid] = x[(idx[c as usize] * d + j) as usize];
    }
}

/// FUSED k-means++ running-min-D² update: per row `i`, the direct squared
/// distance to the sample row `x[center_idx]` (the newly-chosen center, read
/// straight from `x` on the device — no center upload), folded into the
/// running `min_d2` buffer. `first == 1` overwrites (the first center seeds
/// the buffer); otherwise the min is kept. Replaces the per-center GEMM +
/// `row_reduce(Shared)` `distance()` chain in [`kmeanspp_sample`]: one launch
/// + one `n`-float readback per drawn center.
#[cube(launch)]
pub fn kmeanspp_mind2<F: Float + CubeElement>(
    x: &Array<F>,
    min_d2: &mut Array<F>,
    center_idx: u32,
    first: u32,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let xbase = (i as u32) * d;
        let cbase = center_idx * d;
        let mut acc = F::new(0.0_f32);
        let mut j = 0u32;
        while j < d {
            let diff = x[(xbase + j) as usize] - x[(cbase + j) as usize];
            acc += diff * diff;
            j += 1u32;
        }
        if first == 1u32 {
            min_d2[i] = acc;
        } else if acc < min_d2[i] {
            min_d2[i] = acc;
        }
    }
}

/// Per-row squared L2 norm `out[i] = Σ_j X[i,j]²` (one unit per row, ascending
/// `d`-loop). Feeds the GEMM-expansion assignment path: `‖x_i‖²` is computed
/// ONCE per fit and `‖c_j‖²` once per iteration (`n = k` call), then
/// `dist_combine_clamp` forms `max(‖x‖² + ‖c‖² − 2·XCᵀ, 0)` from the matmul
/// cross term. Direct single-loop accumulation — NOT `row_reduce(Shared)` (the
/// PyO3 landmine).
#[cube(launch)]
pub fn row_sqnorm<F: Float + CubeElement>(x: &Array<F>, out: &mut Array<F>, n: u32, d: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = (i as u32) * d;
        let mut acc = F::new(0.0_f32);
        let mut j = 0u32;
        while j < d {
            let v = x[(base + j) as usize];
            acc += v * v;
            j += 1u32;
        }
        out[i] = acc;
    }
}

/// Expand `u32` labels into a row-major `n × k` one-hot `F` matrix (one unit
/// per element). Feeds the GEMM centroid-update path: `sums(k × d) =
/// onehotᵀ · X` runs on the tiled matmul instead of the O(n·k·d) gather.
#[cube(launch)]
pub fn onehot_from_labels<F: Float + CubeElement>(
    labels: &Array<u32>,
    onehot: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let mut v = F::new(0.0_f32);
        if labels[i as usize] == c {
            v = F::new(1.0_f32);
        }
        onehot[tid] = v;
    }
}

/// Row-blocked per-centroid COUNT (no feature dimension — the counts half of
/// [`centroid_sumcount_blocked`] for the GEMM update path, whose sums come
/// from the matmul instead). One unit per `(block, centroid)`.
#[cube(launch)]
pub fn count_blocked(
    labels: &Array<u32>,
    pcounts: &mut Array<u32>,
    n: u32,
    k: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * k;
    if tid < total as usize {
        let b = (tid as u32) / k;
        let c = (tid as u32) % k;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut cnt = 0u32;
        let mut i = start;
        while i < end {
            if labels[i as usize] == c {
                cnt += 1u32;
            }
            i += 1u32;
        }
        pcounts[tid] = cnt;
    }
}

/// Fold the [`count_blocked`] partials: one unit per centroid, ascending scan
/// over the (small) `nblocks` axis.
#[cube(launch)]
pub fn count_reduce(pcounts: &Array<u32>, counts: &mut Array<u32>, k: u32, nblocks: u32) {
    let tid = ABSOLUTE_POS;
    if tid < k as usize {
        let mut cnt = 0u32;
        let mut b = 0u32;
        while b < nblocks {
            cnt += pcounts[(b * k + tid as u32) as usize];
            b += 1u32;
        }
        counts[tid] = cnt;
    }
}

/// Row-blocked per-COLUMN sum over the row-major `x` (`n × d`): one unit per
/// `(block b, column j)`, partials to `psums[b·d + j]`. Stage 1 of the
/// two-pass device `mean(var(X, axis=0))` that scales sklearn's `tol`
/// (Pitfall 6) — the host folds the small `nblocks × d` partials, so the
/// full `n × d` sample matrix is never read back for the tol computation.
#[cube(launch)]
pub fn col_sum_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * d;
    if tid < total as usize {
        let b = (tid as u32) / d;
        let j = (tid as u32) % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            acc += x[(i * d + j) as usize];
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// Row-blocked per-column SQUARED-DEVIATION sum `Σ_i (X[i,j] − means[j])²` —
/// stage 2 of the two-pass device column variance (numerically safe: no
/// `E[x²] − E[x]²` cancellation). Same unit layout as [`col_sum_blocked`].
#[cube(launch)]
pub fn col_sqdiff_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    means: &Array<F>,
    psumsq: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * d;
    if tid < total as usize {
        let b = (tid as u32) / d;
        let j = (tid as u32) % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            let diff = x[(i * d + j) as usize] - means[j as usize];
            acc += diff * diff;
            i += 1u32;
        }
        psumsq[tid] = acc;
    }
}
