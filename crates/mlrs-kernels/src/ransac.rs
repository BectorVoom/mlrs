//! `ransac` — the device kernels behind `RANSACRegressor`'s BATCHED trial scan
//! (RANSAC-02).
//!
//! ## Why RANSAC had no device arm until there was a BATCH
//! [`prims::ransac_host`](mlrs_backend::prims::ransac_host)'s module docs give
//! the reason the engine was host-resident on every backend, and it was a good
//! one: a trial is one `n × d` matvec fused with an `O(n)` threshold, and the
//! loop needs the resulting inlier COUNT on the host before it can draw the next
//! sub-sample, because `max_trials` shrinks as a function of the best consensus
//! found so far. One launch plus one full pipeline stall per trial, a hundred
//! times over, is the launch-latency-bound shape this repo keeps measuring
//! ([[mlrs-gpu-perf-root-cause]]).
//!
//! What changed is not the hardware, it is the observation that **the trials
//! inside a batch do not depend on each other**. A trial's scan reads only the
//! design and that trial's own candidate model; the sequential part — the
//! incumbent comparison, the skip counters, the dynamic `max_trials`, the stop
//! rules — consumes the scans *afterwards*, in trial order. So `B` trials can be
//! drawn, solved and scanned speculatively in ONE launch, and the bookkeeping
//! then replayed over them in order. If it decides to stop at trial `k < B`, the
//! surplus scans are discarded and the draw stream is rewound to where trial `k`
//! left it: the answer is bit-identical to the unbatched loop, and the only cost
//! is the wasted work of `B − k − 1` trials.
//!
//! That turns "one launch and one stall per trial" into "one launch and one
//! stall per batch", which is the only thing that made a device arm worth
//! writing.
//!
//! ## What crosses the bus, and what does not
//! Per batch the host uploads `B·t·d` coefficients and `B·t` intercepts (a few
//! KB) and reads back `B·nblocks·(1 + 2t)` partials. **Nothing of size `n` is
//! ever transferred**: the inlier mask stays device-resident in `mask`, and it
//! is read back exactly once per *improvement* (typically under ten times in a
//! whole fit), not once per trial.
//!
//! The R² the consensus tie-break needs is split across the two kernels here for
//! that same reason. [`ransac_scan_batch`] emits the numerator (`Σ_inliers r²`)
//! and `Σ_inliers y` in the pass it is already making; the host folds the latter
//! into the inlier MEAN and hands it back to [`ransac_den_block`], which forms
//! the two-pass denominator `Σ_inliers (y − ȳ)²` by re-reading the resident
//! mask. Two passes, because the one-pass `Σy² − n·ȳ²` identity is a different
//! sum in floating point and this quantity decides a tie-break
//! ([`ransac_host::r2_on_mask`](mlrs_backend::prims::ransac_host)). The second
//! kernel is `O(n·t)` with no `d` factor and only runs for a trial that has
//! already matched the incumbent's consensus size.
//!
//! ## Blocking, and why the fold order is fixed
//! Both kernels are ROW-BLOCKED with block boundaries that depend only on `n`:
//! unit `(b, blk)` owns rows `[blk·rows_per_block, …)` of trial `b` and writes
//! its own partial slot. The host folds those slots **in block order**, which is
//! what makes the device arm's reduction independent of how the runtime
//! schedules the units — the same property
//! [`ransac_host`](mlrs_backend::prims::ransac_host) buys with fixed
//! `SCAN_BLOCK`s, and it is load-bearing for the same reason (a reassociated sum
//! could flip an R² tie-break and pick a different final model).
//!
//! The count rides the same `F` partial array as the sums rather than a separate
//! `u32` one: a block's count is bounded by `rows_per_block`, which the launcher
//! keeps at `√n`-ish and well inside `f32`'s exact-integer range, so folding it
//! as a float is exact.
//!
//! ## cubecl-cpu MLIR safety
//! House rules kept ([`sgd`](crate::sgd)/[`gmm`](crate::gmm)): only `F`/`u32`
//! accumulators, `if`-guarded forward `while` loops, statement-form `if` (never
//! an `if`-expression in value position), no `SharedMemory`, no `bool`, no
//! infinity sentinel, no scatter beyond the disjoint per-unit slots. That is
//! what lets a direct cubecl-cpu execution test verify kernels whose production
//! backend is a GPU ([[mlrs-gaussian-mixture-cuda-device]]'s technique) — and
//! here it is more than a test convenience, because `device="gpu"` is a legal
//! request on a cpu-backend build and lands on exactly these kernels.
//!
//! Tests live in `crates/mlrs-algos/tests/ransac_device_test.rs` (AGENTS.md §2).

use cubecl::prelude::*;

/// Per-block quantities [`ransac_scan_batch`] emits, as a stride multiplier on
/// the target count: `1 + 2·t` — the inlier COUNT, then `t` squared-error sums,
/// then `t` inlier target sums.
///
/// Exported so the launching prim and the kernel cannot disagree about the
/// layout, the [`HUBER_QUANTITIES`](crate::huber::HUBER_QUANTITIES) precedent.
pub const RANSAC_SCAN_BASE: u32 = 1;

/// Scan the FULL design against `B` candidate models at once, writing each
/// trial's inlier mask and its per-block reduction partials.
///
/// One unit per `(trial b, row-block blk)` at `ABSOLUTE_POS = b·nblocks + blk`.
/// Every unit reads the whole design rows of its block and all `t·d`
/// coefficients of its own trial, and writes only into `mask[b·n + i]` for its
/// rows and `part[(b·nblocks + blk)·stride + …]` — disjoint by construction, so
/// there is no atomic anywhere in this kernel.
///
/// `coef` is `B` consecutive `t × d` row-major blocks (sklearn's `coef_`
/// layout, one per trial) and `icept` is `B` consecutive length-`t` blocks.
/// `squared = 1` selects sklearn's `"squared_error"` loss, `0` its
/// `"absolute_error"` default; both reduce a row's per-target error to the one
/// scalar `residual_threshold` is compared against, summing over the target
/// axis exactly as `np.sum(..., axis=1)` does.
///
/// `part` is laid out `[count, sq_0..sq_{t-1}, ysum_0..ysum_{t-1}]` per slot
/// ([`RANSAC_SCAN_BASE`]). `ysum` is `Σ_{inliers} y` — the inlier MEAN's
/// numerator, gathered here because this pass is already reading `y` and the
/// alternative is a whole extra `O(n·t)` launch before the denominator kernel
/// can run.
///
/// ## The `t == 1` arm is separate, and that is the point
/// A row is an inlier only once its error is summed over ALL targets, but the
/// squared errors may be banked only if it IS one — so the general arm walks the
/// target axis twice, recomputing the dot products for inlier rows. At `t == 1`
/// (every 1-D `y`, i.e. essentially every RANSAC) that recomputation would
/// double the `O(n·d)` cost of the whole engine, so that case gets a body which
/// keeps its single squared error in a register and never re-reads the row.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ransac_scan_batch<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    coef: &Array<F>,
    icept: &Array<F>,
    mask: &mut Array<u32>,
    part: &mut Array<F>,
    n: u32,
    d: u32,
    t: u32,
    batch: u32,
    nblocks: u32,
    rows_per_block: u32,
    threshold: F,
    squared: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = batch * nblocks;
    if tid < total as usize {
        let b = (tid as u32) / nblocks;
        let blk = (tid as u32) % nblocks;
        let start = blk * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let zero = F::new(0.0_f32);
        let stride = RANSAC_SCAN_BASE + 2u32 * t;
        let slot = (tid as u32) * stride;
        let mut count = zero;

        if t == 1u32 {
            let bias = icept[b as usize];
            let cbase = b * d;
            let mut sq_acc = zero;
            let mut ysum = zero;
            let mut i = start;
            while i < end {
                let xbase = i * d;
                let mut dot = zero;
                let mut j = 0u32;
                while j < d {
                    dot += x[(xbase + j) as usize] * coef[(cbase + j) as usize];
                    j += 1u32;
                }
                let yi = y[i as usize];
                let diff = yi - (dot + bias);
                let sq = diff * diff;
                let mut resid = diff.abs();
                if squared == 1u32 {
                    resid = sq;
                }
                let mut flag = 0u32;
                if resid <= threshold {
                    flag = 1u32;
                    count += F::new(1.0_f32);
                    sq_acc += sq;
                    ysum += yi;
                }
                mask[(b * n + i) as usize] = flag;
                i += 1u32;
            }
            part[slot as usize] = count;
            part[(slot + RANSAC_SCAN_BASE) as usize] = sq_acc;
            part[(slot + RANSAC_SCAN_BASE + 1u32) as usize] = ysum;
        } else {
            // Zero the target-indexed accumulator slots before the row loop
            // folds into them; a partial slot is written by exactly this unit,
            // so this is initialization and not a race.
            let mut q = 0u32;
            while q < 2u32 * t {
                part[(slot + RANSAC_SCAN_BASE + q) as usize] = zero;
                q += 1u32;
            }
            let mut i = start;
            while i < end {
                let xbase = i * d;
                let ybase = i * t;
                let mut resid = zero;
                let mut k = 0u32;
                while k < t {
                    let cbase = (b * t + k) * d;
                    let mut dot = zero;
                    let mut j = 0u32;
                    while j < d {
                        dot += x[(xbase + j) as usize] * coef[(cbase + j) as usize];
                        j += 1u32;
                    }
                    let diff = y[(ybase + k) as usize] - (dot + icept[(b * t + k) as usize]);
                    let mut e = diff.abs();
                    if squared == 1u32 {
                        e = diff * diff;
                    }
                    resid += e;
                    k += 1u32;
                }
                let mut flag = 0u32;
                if resid <= threshold {
                    flag = 1u32;
                    count += F::new(1.0_f32);
                    // Second walk of the target axis: the squared errors and the
                    // target sums may only be banked once the row is known to be
                    // an inlier (kernel docs).
                    let mut k2 = 0u32;
                    while k2 < t {
                        let cbase = (b * t + k2) * d;
                        let mut dot = zero;
                        let mut j = 0u32;
                        while j < d {
                            dot += x[(xbase + j) as usize] * coef[(cbase + j) as usize];
                            j += 1u32;
                        }
                        let yv = y[(ybase + k2) as usize];
                        let diff = yv - (dot + icept[(b * t + k2) as usize]);
                        part[(slot + RANSAC_SCAN_BASE + k2) as usize] += diff * diff;
                        part[(slot + RANSAC_SCAN_BASE + t + k2) as usize] += yv;
                        k2 += 1u32;
                    }
                }
                mask[(b * n + i) as usize] = flag;
                i += 1u32;
            }
            part[slot as usize] = count;
        }
    }
}

/// The R² DENOMINATOR of ONE trial: `part[blk·t + k] = Σ_{i ∈ block, inlier}
/// (y[i·t + k] − ymean[k])²`.
///
/// One unit per row-block at `ABSOLUTE_POS = blk`. `mask_off` is `b·n`, the
/// offset of this trial's slice of the resident mask [`ransac_scan_batch`]
/// wrote — so the denominator costs no transfer at all beyond the `t` means
/// going up and the `nblocks·t` partials coming down.
///
/// Launched per trial rather than per batch because it is needed only for a
/// trial that has already matched the incumbent's consensus size, which is a
/// handful of trials in a whole fit (module docs).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ransac_den_block<F: Float + CubeElement>(
    y: &Array<F>,
    mask: &Array<u32>,
    ymean: &Array<F>,
    part: &mut Array<F>,
    n: u32,
    t: u32,
    nblocks: u32,
    rows_per_block: u32,
    mask_off: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nblocks as usize {
        let blk = tid as u32;
        let start = blk * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let zero = F::new(0.0_f32);
        let base = blk * t;
        let mut k0 = 0u32;
        while k0 < t {
            part[(base + k0) as usize] = zero;
            k0 += 1u32;
        }
        let mut i = start;
        while i < end {
            if mask[(mask_off + i) as usize] == 1u32 {
                let ybase = i * t;
                let mut k = 0u32;
                while k < t {
                    let dv = y[(ybase + k) as usize] - ymean[k as usize];
                    part[(base + k) as usize] += dv * dv;
                    k += 1u32;
                }
            }
            i += 1u32;
        }
    }
}
