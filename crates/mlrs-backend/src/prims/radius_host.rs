//! `radius_host` — the HOST arm of `radius_neighbors` (NEIGH-RADIUS-HOST).
//!
//! A fused, worker-pool-parallel `distance → threshold → compact` scan covering
//! every [`Metric`], and — measured, see [`radius_host_applicable`] — the arm
//! that serves EVERY backend by default, not just cpu.
//!
//! ## Why it exists (measured, and it was the one honest regression)
//! `radius_neighbors` shipped (NEIGH-PARAMS) as `metric_distance` into a
//! `rows × n_train` device tile, a readback, and a host scalar threshold scan.
//! That LOST to sklearn 2-3x at every match density — the one estimator surface
//! in the neighbors family that did. sklearn's brute path
//! (`pairwise_distances_chunked` + `np.where`) does the same three things, but
//! its distance block comes out of BLAS and its threshold is one vectorized
//! numpy pass, where mlrs paid an unvectorized element-at-a-time comparison over
//! a block it had just written and read back.
//!
//! The fix is not a faster scan of the tile — it is not writing the tile. This
//! arm keeps a `QB`-wide strip of query rows in registers and thresholds each
//! training row's distances the moment they are computed, so the whole
//! `n_query × n_train` distance block is never materialized anywhere.
//!
//! Wall clock on a 16-core Zen5, `n_train = 20_000`, `n_query = 2_000`,
//! `d = 16`, f32, euclidean, best of 3 interleaved
//! (`scripts/bench_nearest_neighbors_params.py radius`, cpu backend).
//! "tile scan" is `MLRS_RADIUS_HOST=0` — the old composition, measured with the
//! SAME (fixed) egress so the two columns differ only in the scan:
//!
//! | density | tile scan | this arm | sklearn | tile scan | this arm |
//! |---------|-----------|----------|---------|-----------|----------|
//! | 1%      | 0.1732 s  | 0.0101 s | 0.015 s | 0.11x     | **1.53x**|
//! | 5%      | 0.1949 s  | 0.0150 s | 0.024 s | 0.16x     | **1.62x**|
//! | 15%     | 0.2321 s  | 0.0271 s | 0.050 s | 0.20x     | **1.83x**|
//! | 35%     | 0.2791 s  | 0.0444 s | 0.099 s | 0.35x     | **2.22x**|
//! | 60%     | 0.2913 s  | 0.0607 s | 0.121 s | 0.42x     | **1.99x**|
//!
//! ## The OTHER half of that regression was the egress, not the scan
//! Fixing this arm alone did not make the estimator win: at 60% density the
//! Python surface still took 1.74 s, because `radius_neighbors` returned its
//! ~24 MILLION matches to Python as a `list` — one boxed object per element,
//! built serially, then converted back to numpy. Both halves were needed; the
//! egress is now a pyarrow view
//! (`crate::estimators::neighbors::radius_result_to_pyarrow` in `mlrs-py`, which
//! carries that measurement). A parallel scan behind a serial O(matches) egress
//! is invisible — check the boundary before concluding the kernel is the cost.
//!
//! ## It is [`knn_host`]'s scan with a different consumer
//! The per-training-row lane loop — the part that has to vectorize — is
//! [`lane_distances`](super::knn_host::lane_distances), called VERBATIM from
//! both scans, including its AVX2 `#[target_feature]` twin
//! ([`host_simd`](super::host_simd)). What differs is only what happens to the
//! `QB` distances afterwards: `knn_host` folds each into a sorted `k`-list, this
//! one compares against the threshold and appends the survivors. Sharing the
//! loop is what keeps the two from drifting on a metric's clamp, its boundary
//! transform, or a NaN.
//!
//! ## Ordering, and why the lanes get private buffers
//! Each query row's matches must come out in ASCENDING TRAINING INDEX order
//! (sklearn's `sort_results=False` order, which the oracle compares
//! positionally). A block advances all `QB` lanes through the training rows
//! together, so matches for different lanes interleave in TIME — each lane
//! therefore appends into its own buffer, and the block flushes the lanes in
//! order once the training scan is done. Within a lane the appends are already
//! ascending because `t` only increases.
//!
//! ## Threshold in the metric's INTERNAL units (Pitfall 8)
//! The scan compares the pre-boundary-transform value against the transformed
//! radius — `radius²` for Euclidean, `radius^p` for Minkowski, `radius` for the
//! rest — and applies the root only to the matches. Both transforms are monotone
//! on `[0, ∞)`, so this selects exactly the set the true distances would; the
//! square root is then paid `count` times instead of `n_query · n_train` times.
//!
//! Tests live in `crates/mlrs-backend/tests/radius_scan_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_core::PrimError;

use super::hgb_host::HostFloat;
use super::host_pool::{Shared, WorkerPool};
use super::host_simd::avx2_available;
use super::knn_graph::Metric;
use super::knn_host::{
    host_units, lane_distances, minkowski_powi, row_inv_norm_host, stage_tile, MID_CHEBYSHEV,
    MID_COSINE, MID_EUCLIDEAN, MID_MANHATTAN, MID_MINKOWSKI, QB,
};
use super::radius::RadiusMatches;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Should the host arm serve this scan?
///
/// **True on every backend by default.** On cpu that is obvious — the device
/// composition materializes a tile in host memory only to read it back and scan
/// it. On the GPU backends it is a measurement: this arm is 3-6x faster than the
/// device count+compaction engine on both of them, and the only arm that beats
/// sklearn at all (the table lives on
/// [`radius_device_applicable`](super::radius::radius_device_applicable), with
/// the reason a shared-memory device has nothing to save, and the caveat that a
/// discrete GPU across PCIe is untested here).
///
/// `MLRS_RADIUS_DEVICE=1` stands this arm down in favour of the device engine —
/// that is the A/B the table was produced with, and the switch a discrete-GPU
/// deployment would flip. `MLRS_RADIUS_HOST=0` stands it down in favour of the
/// original tile-readback scan (the "before" column); `MLRS_RADIUS_HOST=1`
/// forces it even when the device engine is on.
pub fn radius_host_applicable(n_query: usize, n_train: usize, n_features: usize) -> bool {
    if !(n_query > 0 && n_train > 0 && n_features > 0) {
        return false;
    }
    if let Some(v) = crate::abflag::var("MLRS_RADIUS_HOST") {
        return v != "0";
    }
    // An explicit device-engine force wins over this arm's default.
    !crate::abflag::var("MLRS_RADIUS_DEVICE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Every training point within `radius` of each query row, under `metric`,
/// computed on the HOST.
///
/// Returns the same [`RadiusMatches`] layout the device arm returns — per-row
/// matches concatenated in ascending training-index order, plus the per-row
/// counts — so the dispatch site does not know which arm ran.
///
/// Geometry and the untrusted `radius` are validated here rather than assumed:
/// this is a `pub` prim and the scan indexes `xq[q*d + c]` / `xt[t*d + c]`
/// directly.
pub fn radius_host_scan<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, n_features): (usize, usize),
    x_train: &DeviceArray<ActiveRuntime, F>,
    n_train: usize,
    radius: f64,
    metric: Metric,
) -> Result<RadiusMatches<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: every bound the scan indexes with. ---
    if n_query
        .checked_mul(n_features)
        .map(|v| v != xq.len())
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_query,
            cols: n_features,
            len: xq.len(),
        });
    }
    if n_train
        .checked_mul(n_features)
        .map(|v| v != x_train.len())
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: n_features,
            len: x_train.len(),
        });
    }
    if n_train > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "rows",
            rows: n_train,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    // A negative or NaN radius is the caller's contract violation; the estimator
    // rejects it earlier with its own error, this is the prim-level backstop.
    if !(radius >= 0.0) {
        return Err(PrimError::ShapeMismatch {
            operand: "radius",
            rows: 1,
            cols: 0,
            len: 0,
        });
    }

    // On the cpu backend a read-back is a host memcpy over `n × d`, amortized
    // over an `n_query × n_train × n_features` scan — the same trade
    // `knn_host_topk` makes, and the block this arm does NOT materialize is
    // `n_query × n_train`, which is larger by `n_train/d`.
    let xq_h: Vec<F> = xq.to_host(pool);
    let xt_h: Vec<F> = x_train.to_host(pool);

    let (distances, indices, counts) = if size_of::<F>() == 4 {
        let (v, i, c) = scan_typed::<f32>(
            bytemuck::cast_slice(&xq_h),
            bytemuck::cast_slice(&xt_h),
            n_query,
            n_features,
            n_train,
            radius,
            metric,
        );
        (bytemuck::cast_slice::<f32, F>(&v).to_vec(), i, c)
    } else {
        let (v, i, c) = scan_typed::<f64>(
            bytemuck::cast_slice(&xq_h),
            bytemuck::cast_slice(&xt_h),
            n_query,
            n_features,
            n_train,
            radius,
            metric,
        );
        (bytemuck::cast_slice::<f64, F>(&v).to_vec(), i, c)
    };

    Ok(RadiusMatches {
        distances,
        indices,
        counts,
    })
}

/// Dispatch the metric ONCE, outside the scan, into a monomorphized loop —
/// [`knn_host`](super::knn_host)'s `topk_typed`, with the threshold consumer.
///
/// The threshold is pre-transformed into the metric's internal units here, which
/// is the only place that knows both `p` and which `MID` was selected.
fn scan_typed<T: HostFloat>(
    xq: &[T],
    xt: &[T],
    n_query: usize,
    d: usize,
    n_train: usize,
    radius: f64,
    metric: Metric,
) -> (Vec<T>, Vec<i32>, Vec<u32>) {
    match metric {
        // `radius²`: the Euclidean lane loop accumulates the SQUARED distance.
        Metric::Euclidean => {
            scan::<T, MID_EUCLIDEAN, 0>(xq, xt, n_query, d, n_train, radius * radius, 2.0)
        }
        Metric::Manhattan => scan::<T, MID_MANHATTAN, 0>(xq, xt, n_query, d, n_train, radius, 1.0),
        Metric::Chebyshev => scan::<T, MID_CHEBYSHEV, 0>(xq, xt, n_query, d, n_train, radius, 1.0),
        Metric::Cosine => scan::<T, MID_COSINE, 0>(xq, xt, n_query, d, n_train, radius, 1.0),
        // `radius^p`: the Minkowski lane loop accumulates `Σ|Δ|^p`. Same integer-
        // exponent specialization `knn_host` documents (a `powf` per feature is
        // a scalar libm call that blocks the lane loop's vectorization).
        Metric::Minkowski { p } => {
            let thresh = radius.powf(p);
            match minkowski_powi(p) {
                Some(3) => scan::<T, MID_MINKOWSKI, 3>(xq, xt, n_query, d, n_train, thresh, p),
                Some(4) => scan::<T, MID_MINKOWSKI, 4>(xq, xt, n_query, d, n_train, thresh, p),
                Some(5) => scan::<T, MID_MINKOWSKI, 5>(xq, xt, n_query, d, n_train, thresh, p),
                Some(6) => scan::<T, MID_MINKOWSKI, 6>(xq, xt, n_query, d, n_train, thresh, p),
                Some(7) => scan::<T, MID_MINKOWSKI, 7>(xq, xt, n_query, d, n_train, thresh, p),
                Some(8) => scan::<T, MID_MINKOWSKI, 8>(xq, xt, n_query, d, n_train, thresh, p),
                _ => scan::<T, MID_MINKOWSKI, 0>(xq, xt, n_query, d, n_train, thresh, p),
            }
        }
    }
}

/// One worker's slice of the ragged result: its query rows' matches, already
/// concatenated in row order.
///
/// The workers own DISJOINT, CONTIGUOUS ranges of query blocks, so concatenating
/// the slices in worker order reproduces the global row-major layout — there is
/// no merge step and no shared output buffer to synchronize on.
struct WorkerSlice<T> {
    distances: Vec<T>,
    indices: Vec<i32>,
    counts: Vec<u32>,
}

impl<T> Default for WorkerSlice<T> {
    fn default() -> Self {
        Self {
            distances: Vec::new(),
            indices: Vec::new(),
            counts: Vec::new(),
        }
    }
}

/// The scan itself: `n_query × n_train` distances under metric `MID`, reduced to
/// the matches within `thresh` (already in the metric's internal units).
///
/// Parallel over contiguous ranges of query BLOCKS, exactly as
/// [`knn_host`](super::knn_host)'s `scan` is — the only synchronization in the
/// whole search is the pool's two barrier crossings.
fn scan<T: HostFloat, const MID: u32, const PI: u32>(
    xq: &[T],
    xt: &[T],
    n_query: usize,
    d: usize,
    n_train: usize,
    thresh: f64,
    p: f64,
) -> (Vec<T>, Vec<i32>, Vec<u32>) {
    // Cosine's two reciprocal-norm vectors are hoisted out of the scan, as in
    // the k-nearest arm (its lane loop is shared with this one).
    let (xnorm, ynorm) = if MID == MID_COSINE {
        (
            row_inv_norm_host(xq, n_query, d),
            row_inv_norm_host(xt, n_train, d),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let units = host_units(n_query);
    let n_blocks = n_query.div_ceil(QB);
    let mut slices: Vec<WorkerSlice<T>> = (0..units).map(|_| WorkerSlice::default()).collect();

    {
        let sh = Shared::new(&mut slices);
        let pool = WorkerPool::new(units);
        let pass = |u: usize| {
            let lo = n_blocks * u / units;
            let hi = n_blocks * (u + 1) / units;
            // SAFETY: worker `u` touches ONLY `slices[u]` (the `Shared`
            // contract) — the slices are per-worker outputs, not a partitioned
            // view of one buffer.
            let slice = &mut unsafe { sh.get_mut() }[u];
            if lo >= hi {
                return;
            }

            // Per-worker scratch, allocated ONCE for the whole range: the
            // transposed query tile, the cosine lane norms, and the QB private
            // match buffers the block flushes in lane order.
            let mut xtile: Vec<[T; QB]> = vec![[T::ZERO; QB]; d];
            let mut xn = [T::ZERO; QB];
            let mut lane_val: Vec<Vec<T>> = (0..QB).map(|_| Vec::new()).collect();
            let mut lane_idx: Vec<Vec<u32>> = (0..QB).map(|_| Vec::new()).collect();

            for b in lo..hi {
                let q0 = b * QB;
                let active = (n_query - q0).min(QB);

                stage_tile(xq, &mut xtile, q0, active, d);
                if MID == MID_COSINE {
                    for (a, slot) in xn.iter_mut().enumerate() {
                        *slot = if a < active { xnorm[q0 + a] } else { T::ZERO };
                    }
                }
                for a in 0..QB {
                    lane_val[a].clear();
                    lane_idx[a].clear();
                }

                dispatch_threshold_block::<T, MID, PI>(
                    xt,
                    &xtile,
                    &xn,
                    &ynorm,
                    n_train,
                    d,
                    active,
                    thresh,
                    p,
                    &mut lane_val,
                    &mut lane_idx,
                );

                // Flush the block's lanes IN ORDER: lane `a` is query row
                // `q0 + a`, and within a lane the appends are already ascending
                // by training index.
                emit(
                    &lane_val,
                    &lane_idx,
                    active,
                    p,
                    &mut slice.distances,
                    &mut slice.indices,
                    &mut slice.counts,
                );
            }
        };
        pool.run(&pass);
    }

    let total: usize = slices.iter().map(|s| s.distances.len()).sum();
    let mut distances: Vec<T> = Vec::with_capacity(total);
    let mut indices: Vec<i32> = Vec::with_capacity(total);
    let mut counts: Vec<u32> = Vec::with_capacity(n_query);
    for s in slices {
        distances.extend_from_slice(&s.distances);
        indices.extend_from_slice(&s.indices);
        counts.extend_from_slice(&s.counts);
    }
    (distances, indices, counts)
}

/// Run one block's threshold scan through the WIDEST vector unit this machine
/// actually has, instead of the one the crate was compiled for.
///
/// The rationale is [`host_simd`](super::host_simd)'s, and this is the same
/// explicit-twin shape [`knn_host`](super::knn_host)'s `dispatch_scan_block`
/// uses; `radius_scan_test::avx2_and_baseline_agree_bitwise` is the identity
/// gate.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dispatch_threshold_block<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    active: usize,
    thresh: f64,
    p: f64,
    lane_val: &mut [Vec<T>],
    lane_idx: &mut [Vec<u32>],
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: guarded by the runtime detection this branch tests, and the
        // body is the ordinary `threshold_block` — nothing in it is unsafe.
        unsafe {
            threshold_block_avx2::<T, MID, PI>(
                xt, xtile, xn, ynorm, n_train, d, active, thresh, p, lane_val, lane_idx,
            );
        }
        return;
    }
    threshold_block::<T, MID, PI>(
        xt, xtile, xn, ynorm, n_train, d, active, thresh, p, lane_val, lane_idx,
    );
}

/// [`threshold_block`] compiled for AVX2 + FMA — see [`dispatch_threshold_block`].
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`
/// ([`avx2_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn threshold_block_avx2<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    active: usize,
    thresh: f64,
    p: f64,
    lane_val: &mut [Vec<T>],
    lane_idx: &mut [Vec<u32>],
) {
    threshold_block::<T, MID, PI>(
        xt, xtile, xn, ynorm, n_train, d, active, thresh, p, lane_val, lane_idx,
    );
}

/// One block's full training scan: every training row, once, thresholded.
///
/// The `QB` distances come from the SHARED lane loop
/// ([`lane_distances`](super::knn_host::lane_distances)); this adds the
/// block-wide screen and the per-lane append.
///
/// ## The screen
/// One `hit` pass over the lanes before any append, for the same reason the
/// k-nearest arm screens before any insert: at a useful radius the overwhelming
/// majority of training rows match NO lane in the block, and proving that with
/// `QB` compares (which vectorize) is much cheaper than `QB` branches into
/// `Vec::push` (which do not). At high density the screen always passes and
/// costs one extra pass over data already in registers.
///
/// A NaN distance appends nothing: `v <= thresh` is false for NaN, which is the
/// same rejection sklearn's `np.where(D <= radius)` performs.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn threshold_block<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    active: usize,
    thresh: f64,
    p: f64,
    lane_val: &mut [Vec<T>],
    lane_idx: &mut [Vec<u32>],
) {
    let pe = T::from_f64(p);
    let th = T::from_f64(thresh);

    for t in 0..n_train {
        let yrow = &xt[t * d..t * d + d];
        let yn = if MID == MID_COSINE { ynorm[t] } else { T::ZERO };
        let acc = lane_distances::<T, MID, PI>(xtile, yrow, xn, yn, pe);

        let mut hit = false;
        for a in 0..active {
            hit |= acc[a] <= th;
        }
        if hit {
            for a in 0..active {
                if acc[a] <= th {
                    lane_val[a].push(acc[a]);
                    lane_idx[a].push(t as u32);
                }
            }
        }
    }
}

/// Append one block's lanes to the worker's slice, applying the metric's
/// BOUNDARY transform to the values it kept.
///
/// The root is applied here — to the matches — and not to any of the
/// `n_train` candidates it was compared against, because it is monotone and so
/// cannot change WHICH candidates matched. That is the same deferral
/// [`knn_host`](super::knn_host)'s `emit` performs for the selected `k`.
fn emit<T: HostFloat>(
    lane_val: &[Vec<T>],
    lane_idx: &[Vec<u32>],
    active: usize,
    p: f64,
    out_val: &mut Vec<T>,
    out_idx: &mut Vec<i32>,
    out_counts: &mut Vec<u32>,
) {
    let inv_p = T::from_f64(1.0 / p);
    for a in 0..active {
        let vals = &lane_val[a];
        for (&v, &ix) in vals.iter().zip(lane_idx[a].iter()) {
            out_val.push(transform::<T>(v, p, inv_p));
            out_idx.push(ix as i32);
        }
        out_counts.push(vals.len() as u32);
    }
}

/// The boundary transform, selected by `p` the way the lane loop's `MID`
/// selects the accumulation: `p = 2` is Euclidean (root), any other `p` is
/// Minkowski (`p`-th root), and the metrics whose lane loop already emits the
/// true distance pass `p = 1`, whose `inv_p` root is the identity.
///
/// Written as one function rather than a `MID` branch so `emit` stays
/// monomorphization-free — it runs once per BLOCK, not once per candidate.
#[inline]
fn transform<T: HostFloat>(v: T, p: f64, inv_p: T) -> T {
    if p == 1.0 {
        v
    } else if p == 2.0 {
        v.sqrt()
    } else {
        v.powf(inv_p)
    }
}
