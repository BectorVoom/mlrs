//! `knn_host` — the HOST arm of the brute-force k-nearest search (KNN-HOST).
//!
//! A plain-Rust, worker-pool-parallel `distance → top-k` scan covering EVERY
//! [`Metric`], any `k` and any `n_features`, used on the cpu backend in place of
//! the device pipeline.
//!
//! ## Why it exists (measured, not assumed)
//! The cpu backend already had a tuned device kernel for the search
//! ([`cpu_rows_topk`](super::knn::cpu_rows_topk)), but its local caches cap it at
//! `k <= 16` / `n_features <= 32` and it implements EUCLIDEAN only. Everything
//! outside that rectangle fell through to `distance → top_k`, which is the
//! GPU-shaped composition — a materialized `tile × n_train` distance matrix plus
//! a shared-memory selection whose `sync_cube` barriers spin across one OS thread
//! per unit on `cubecl-cpu`.
//!
//! `predict` on a 16-core Zen5 at `n_train = 50_000`, `n_query = 5_000`,
//! `d = 16`, against sklearn's `algorithm='brute'` (best of 5, interleaved;
//! `MLRS_KNN_HOST=0` reproduces the "before" column exactly):
//!
//! | config                | before  | after   | sklearn | before | after |
//! |-----------------------|---------|---------|---------|--------|-------|
//! | euclidean, `k = 15`   | 0.034 s | 0.034 s | 0.147 s | 4.3×   | 4.3×  |
//! | euclidean, `k = 20`   | 6.59 s  | 0.042 s | 0.152 s | 0.02×  | 3.6×  |
//! | euclidean, `k = 50`   | 15.0 s  | 0.057 s | 0.208 s | 0.01×  | 3.7×  |
//! | euclidean, `k = 100`  | 29.0 s  | 0.099 s | 0.303 s | 0.01×  | 3.1×  |
//!
//! `k = 15` is unchanged because it is INSIDE the tuned kernel's cap and still
//! dispatches there (see `neighbors::nearest::metric_topk`). `k = 20` is not:
//! one step past the kernel's 16-slot list, an ordinary `n_neighbors` value took
//! the estimator from beating sklearn 4× to losing to it 50×. The cliff, not a
//! slope, was the entire loss.
//!
//! Every non-default `metric` lost by construction, because none of them ever
//! reached the tuned kernel at any `k` — the whole metric surface was served by
//! that same fallback. Same size, `k = 5`:
//!
//! | metric          | before  | after   | sklearn | before | after |
//! |-----------------|---------|---------|---------|--------|-------|
//! | manhattan       | 2.42 s  | 0.041 s | 0.59 s  | 0.20×  | 14×   |
//! | chebyshev       | 2.37 s  | 0.040 s | 0.60 s  | 0.20×  | 15×   |
//! | cosine          | 2.43 s  | 0.058 s | 1.63 s  | 0.64×  | 28×   |
//! | minkowski `p=3` | 5.58 s  | 0.051 s | 7.63 s  | 1.37×  | 150×  |
//!
//! (Minkowski "won" before only because sklearn's own `p != 1, 2` path calls
//! `pow` per feature and is slower still; 1.37× against a 7.6-second baseline is
//! not a win worth keeping.)
//!
//! ## This is now the FALLBACK, not the primary cpu path (KNN-CUBE-METRIC)
//! This module originally served every metric on the cpu backend, on the
//! reasoning that `cubecl-cpu` JITs at LLVM `-O0` and so could not vectorize.
//! That reasoning was wrong in the part that matters: `cubecl-cpu` lowers
//! `Vector<F, Const<N>>` to genuine MLIR `vector<Nxf32>` ops, and it JITs for the
//! HOST cpu — so a CubeCL kernel gets the machine's real register width without
//! anyone asking, which is exactly why the tuned Euclidean kernel outran this
//! arm even after it was given AVX2 by hand.
//!
//! [`cpu_metric_rows_topk`](super::knn::cpu_metric_rows_topk) is that family
//! written for all five metrics, and it is the arm the dispatch now prefers. It
//! matches or beats this one everywhere (50_000 × 5_000 × 16, `k = 5`, best of
//! 5): euclidean 0.028 s vs 0.028, manhattan 0.032 vs 0.037, chebyshev 0.031 vs
//! 0.037, cosine 0.028 vs 0.055, minkowski `p=3` 0.049 vs 0.048.
//!
//! What is left for this module is the rectangle the kernels' COMPTIME caps
//! cannot cover — `k > CPU_METRIC_K_CAP` (128) or `n_features >
//! CPU_METRIC_MAX_COLS` (128) — where the alternative is once again the
//! GPU-shaped `distance → top_k` composition. Those caps are local-array sizes,
//! so lifting them further is a stack decision, not an algorithmic one; this arm
//! has no caps at all and needs none.
//!
//! ## Shape — the same one the tuned kernel uses, for the same reason
//! Rust will not reassociate float arithmetic, so a `Σ_c` reduction over
//! FEATURES cannot vectorize. The lanes therefore have to be QUERY ROWS: a block
//! of [`QB`] query rows is staged into a TRANSPOSED tile (`xt[c][a]` = feature
//! `c` of the block's query row `a`), and the per-training-row step accumulates
//! `QB` INDEPENDENT reductions at once — which LLVM does vectorize, because each
//! lane is its own accumulator. Every training row is read once per block and
//! feeds `QB` query rows, which is also what keeps the scan off the memory bus.
//!
//! ## Result contract (identical to the device arms)
//! The emitted top-k is the `k` smallest `(value, index)` pairs in ascending pair
//! order — the same total order and the same lowest-index tie-break
//! [`insert_lane`](mlrs_kernels::knn) applies, so a query equidistant from two
//! training points resolves to the same neighbour whichever arm ran. Distances
//! are TRUE metric distances (roots applied), so a `weights='distance'` consumer
//! can divide by them directly.
//!
//! Values are NOT bitwise identical to the device arms and are not asserted to
//! be: the Euclidean device kernel accumulates with `fma` (one rounding) where
//! this arm uses `mul` + `add` (two), and the Minkowski root is applied to the
//! `k` selected values here rather than to every pair. Both are within the
//! repo's 1e-5 oracle band, and the SELECTED SET is unaffected — a root is
//! monotone, and `fma` moves the sum toward the exact answer, not away.
//!
//! Tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;

use super::hgb_host::HostFloat;
use super::host_pool::{Shared, WorkerPool};
use super::host_simd::avx2_available;
use super::knn_graph::Metric;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Query rows one worker handles at a time — the SIMD lane axis of the scan.
///
/// 16 is one AVX-512 `f32` register (and four AVX2 ones), and it is also the
/// reuse factor of the training stream: every training row a worker loads feeds
/// this many query rows, so the per-row load is amortized 16 ways.
///
/// It is deliberately the same width the device cpu kernel's expansion arms use
/// ([`CPU_QUERY_TILE`](mlrs_kernels::knn::CPU_QUERY_TILE)); unlike that kernel,
/// nothing here is unrolled at JIT time, so the width costs no compile time and
/// is not the scheduling grain either (the pool splits BLOCKS, and a partial
/// trailing block still runs).
///
/// ## 16 and not 32 (measured, and not unanimous)
/// A 32-wide build was made and benched against this one on a 16-core Zen5 at
/// `n_train = 50_000`, `n_query = 5_000`, `d = 16`, `k = 5` (best of 7,
/// reproduced across two alternating runs):
///
/// | metric        | QB = 16 | QB = 32 |
/// |---------------|---------|---------|
/// | euclidean     | 0.037 s | 0.049 s |
/// | manhattan     | 0.034 s | 0.053 s |
/// | chebyshev     | 0.034 s | 0.034 s |
/// | minkowski p=3 | 0.050 s | 0.048 s |
/// | cosine        | 0.058 s | 0.032 s |
///
/// So 16 wins the default metric and Manhattan, ties the rest, and loses Cosine
/// — whose lane loop is a bare dot product and therefore the one most starved of
/// independent accumulators. Making the width a fourth const generic would let
/// each metric take its own, at the cost of doubling the instantiations of the
/// whole scan; Cosine already beats sklearn 28× at this width, so the split is
/// not worth its weight. Revisit it if the Cosine cell ever becomes the one that
/// matters.
pub(super) const QB: usize = 16;

/// `MID` for [`Metric::Euclidean`] — `Σ (x−y)²`, square root at the boundary.
pub(super) const MID_EUCLIDEAN: u32 = 0;
/// `MID` for [`Metric::Manhattan`] — `Σ |x−y|`, no boundary transform.
pub(super) const MID_MANHATTAN: u32 = 1;
/// `MID` for [`Metric::Chebyshev`] — `max |x−y|`, no boundary transform.
pub(super) const MID_CHEBYSHEV: u32 = 2;
/// `MID` for [`Metric::Minkowski`] — `Σ |x−y|^p`, `p`-th root at the boundary.
pub(super) const MID_MINKOWSKI: u32 = 3;
/// `MID` for [`Metric::Cosine`] — `clamp(1 − x·y/‖x‖‖y‖, 0, 2)`, no transform.
pub(super) const MID_COSINE: u32 = 4;

/// Should the host arm serve this search?
///
/// True on the **cpu** backend, where the device pipeline is structurally wrong
/// for everything the tuned row-scan kernel does not cover (module docs). False
/// on every real device backend, whose kernels are the point of the project —
/// this arm would drag their operands back across the bus.
///
/// `tuned_arm_available` is the caller's answer to "does the measured-faster
/// [`cpu_rows_topk`](super::knn::cpu_rows_topk) kernel cover this exact shape?"
/// (Euclidean, `k <= 16`, `n_features <= 32`). Where it does, it wins by ~2×
/// and this arm stands down — the caller documents that A/B. The question is
/// asked rather than answered here because the answer is a property of the
/// dispatch site's metric, which this prim does not otherwise need.
///
/// `MLRS_KNN_HOST=1` forces the host arm on REGARDLESS of `tuned_arm_available`
/// — that is exactly the A/B the caller's table was measured with;
/// `MLRS_KNN_HOST=0` forces it off, restoring the pre-KNN-HOST dispatch.
pub fn knn_host_applicable(
    n_query: usize,
    n_train: usize,
    n_features: usize,
    k: usize,
    tuned_arm_available: bool,
) -> bool {
    if let Some(v) = crate::abflag::var("MLRS_KNN_HOST") {
        return v != "0";
    }
    if crate::capability::active_backend_name() != "cpu" || tuned_arm_available {
        return false;
    }
    n_query > 0 && n_train > 0 && n_features > 0 && k >= 1 && k <= n_train
}

/// Brute-force `k`-nearest search under `metric`, computed on the HOST.
///
/// Returns `(values, indices)` as freshly uploaded `n_query × k` device arrays —
/// the same pair, in the same order, the device pipeline returns, so callers
/// dispatch between the two without knowing which ran.
///
/// Geometry is validated here rather than assumed: this is a `pub` prim, and the
/// scan indexes `x[q*d + c]` / `y[t*d + c]` directly.
pub fn knn_host_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, n_features): (usize, usize),
    x_train: &DeviceArray<ActiveRuntime, F>,
    n_train: usize,
    k: usize,
    metric: Metric,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, u32>,
    ),
    PrimError,
>
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
    if k < 1 || k > n_train {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: n_train,
        });
    }
    if n_query > u32::MAX as usize || n_train > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "rows",
            rows: n_query.max(n_train),
            cols: 0,
            len: u32::MAX as usize,
        });
    }

    // On the cpu backend a read-back is a host memcpy, and it is amortized over
    // an `n_query × n_train × n_features` scan — at the smallest rung that ships
    // (2_000 × 20_000 × 16) the two copies are under a thousandth of the work.
    let xq_h: Vec<F> = xq.to_host(pool);
    let xt_h: Vec<F> = x_train.to_host(pool);

    let (val, idx) = if size_of::<F>() == 4 {
        let (v, i) = topk_typed::<f32>(
            bytemuck::cast_slice(&xq_h),
            bytemuck::cast_slice(&xt_h),
            n_query,
            n_features,
            n_train,
            k,
            metric,
        );
        (bytemuck::cast_slice::<f32, F>(&v).to_vec(), i)
    } else {
        let (v, i) = topk_typed::<f64>(
            bytemuck::cast_slice(&xq_h),
            bytemuck::cast_slice(&xt_h),
            n_query,
            n_features,
            n_train,
            k,
            metric,
        );
        (bytemuck::cast_slice::<f64, F>(&v).to_vec(), i)
    };

    Ok((
        DeviceArray::from_host(pool, &val),
        DeviceArray::from_host(pool, &idx),
    ))
}

/// Dispatch the metric ONCE, outside the scan, into a monomorphized loop.
///
/// The device kernels select their distance form from a runtime scalar on every
/// pair; on the host that switch would sit in the innermost loop and block
/// vectorization outright. Lifting it to a const generic makes each instantiated
/// inner step a straight-line expression — the host twin of specializing the
/// shader (the [`sgd_host`](super::sgd_host) precedent).
fn topk_typed<T: HostFloat>(
    xq: &[T],
    xt: &[T],
    n_query: usize,
    d: usize,
    n_train: usize,
    k: usize,
    metric: Metric,
) -> (Vec<T>, Vec<u32>) {
    match metric {
        Metric::Euclidean => scan::<T, MID_EUCLIDEAN, 0>(xq, xt, n_query, d, n_train, k, 2.0),
        Metric::Manhattan => scan::<T, MID_MANHATTAN, 0>(xq, xt, n_query, d, n_train, k, 1.0),
        Metric::Chebyshev => scan::<T, MID_CHEBYSHEV, 0>(xq, xt, n_query, d, n_train, k, 1.0),
        Metric::Cosine => scan::<T, MID_COSINE, 0>(xq, xt, n_query, d, n_train, k, 1.0),
        // An INTEGER exponent is raised by repeated multiplication instead of
        // `powf`, which is the difference between a vectorized lane loop and a
        // scalar libm call per feature — measured 20x on the whole search (see
        // `MINKOWSKI_POWI_MAX`). `p = 1` / `p = 2` never reach here: the Python
        // shim collapses them onto Manhattan / Euclidean the way sklearn's
        // `_check_algorithm_metric` does.
        Metric::Minkowski { p } => match minkowski_powi(p) {
            Some(3) => scan::<T, MID_MINKOWSKI, 3>(xq, xt, n_query, d, n_train, k, p),
            Some(4) => scan::<T, MID_MINKOWSKI, 4>(xq, xt, n_query, d, n_train, k, p),
            Some(5) => scan::<T, MID_MINKOWSKI, 5>(xq, xt, n_query, d, n_train, k, p),
            Some(6) => scan::<T, MID_MINKOWSKI, 6>(xq, xt, n_query, d, n_train, k, p),
            Some(7) => scan::<T, MID_MINKOWSKI, 7>(xq, xt, n_query, d, n_train, k, p),
            Some(8) => scan::<T, MID_MINKOWSKI, 8>(xq, xt, n_query, d, n_train, k, p),
            _ => scan::<T, MID_MINKOWSKI, 0>(xq, xt, n_query, d, n_train, k, p),
        },
    }
}

/// Largest Minkowski exponent served by the repeated-multiplication lane loop.
///
/// Above it (and for any non-integer `p`) the scan falls back to `powf`, which
/// is correct for every `p >= 1` and merely slower. The cap is where the unrolled
/// multiply chain stops being obviously cheaper than one libm call, and it covers
/// every exponent a user plausibly types.
const MINKOWSKI_POWI_MAX: u32 = 8;

/// The integer exponent `p` denotes, when it is exactly one this scan
/// specializes for.
///
/// `p` must round-trip through `u32` EXACTLY — `p = 3.0000001` is a different
/// metric from `p = 3` and must not silently take the integer path.
pub(super) fn minkowski_powi(p: f64) -> Option<u32> {
    let pi = p as u32;
    if p == f64::from(pi) && (3..=MINKOWSKI_POWI_MAX).contains(&pi) {
        Some(pi)
    } else {
        None
    }
}

/// Per-row RECIPROCAL norms `1/‖v[r]‖`, computed only for [`MID_COSINE`].
///
/// The device kernel ([`cosine_dist`](mlrs_kernels::distance::cosine_dist))
/// keeps SQUARED norms and forms `dot / sqrt(‖x‖²·‖y‖²)` per pair. Reciprocals
/// are the same value reassociated into two multiplies, and here that is the
/// difference between a vectorized epilogue and a scalar one: the per-pair
/// `sqrt` + divide measured as MORE than the whole dot loop (cosine ran 2.3×
/// slower than Euclidean at `50_000 × 5_000 × 16` despite a strictly cheaper
/// inner step, and folding them into this pass removed all of it).
///
/// A zero-norm row gets `0`, so its similarity is `0` and its distance `1` —
/// the same answer the kernel's `if denom > 0` guard produces.
///
/// This is also the form sklearn uses: `cosine_distances` normalizes both
/// operands and then dots them, rather than dividing the dot afterwards.
pub(super) fn row_inv_norm_host<T: HostFloat>(v: &[T], rows: usize, d: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; rows];
    for (r, o) in out.iter_mut().enumerate() {
        let row = &v[r * d..(r + 1) * d];
        let mut acc = T::ZERO;
        for &e in row {
            acc = acc + e * e;
        }
        *o = if acc > T::ZERO {
            T::ONE / acc.sqrt()
        } else {
            T::ZERO
        };
    }
    out
}

/// Worker count for one search.
///
/// Bounded by the number of query BLOCKS so a small query set does not spawn
/// workers that would own nothing and only pay barrier crossings, and by the
/// machine's own unit count ([`cpu_launch_units`](crate::capability::cpu_launch_units),
/// which `MLRS_CPU_UNITS` overrides for A/B).
pub(super) fn host_units(n_query: usize) -> usize {
    let blocks = n_query.div_ceil(QB);
    blocks
        .max(1)
        .min(crate::capability::cpu_launch_units().max(1) as usize)
}

/// The scan itself: `n_query × n_train` distances under metric `MID`, reduced to
/// the `k` nearest per query row.
///
/// Parallel over contiguous ranges of query BLOCKS — each worker owns a disjoint
/// set of output rows, so the only synchronization in the whole search is the
/// pool's two barrier crossings.
fn scan<T: HostFloat, const MID: u32, const PI: u32>(
    xq: &[T],
    xt: &[T],
    n_query: usize,
    d: usize,
    n_train: usize,
    k: usize,
    p: f64,
) -> (Vec<T>, Vec<u32>) {
    // Cosine is the one metric whose pair value is not a function of the feature
    // differences alone; its two norm vectors are hoisted out of the scan exactly
    // as the device path hoists them into `row_sumsq` feeders.
    let (xnorm, ynorm) = if MID == MID_COSINE {
        (
            row_inv_norm_host(xq, n_query, d),
            row_inv_norm_host(xt, n_train, d),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let mut out_val = vec![T::ZERO; n_query * k];
    let mut out_idx = vec![0u32; n_query * k];
    let units = host_units(n_query);
    let n_blocks = n_query.div_ceil(QB);

    {
        let sh_val = Shared::new(&mut out_val);
        let sh_idx = Shared::new(&mut out_idx);
        let pool = WorkerPool::new(units);
        let pass = |u: usize| {
            let lo = n_blocks * u / units;
            let hi = n_blocks * (u + 1) / units;
            if lo >= hi {
                return;
            }
            // SAFETY: block `b` owns output rows `[b*QB, min((b+1)*QB, n_query))`
            // and this worker owns blocks `[lo, hi)` exclusively, so no two
            // workers write the same element (the `Shared` contract).
            let out_val = unsafe { sh_val.get_mut() };
            let out_idx = unsafe { sh_idx.get_mut() };

            // Per-worker scratch, allocated ONCE for the whole range: the
            // transposed query tile, the QB independent accumulators, and the QB
            // sorted top-k lists with their worst-kept mirrors.
            let mut xtile: Vec<[T; QB]> = vec![[T::ZERO; QB]; d];
            let mut lval = vec![T::ZERO; QB * k];
            let mut lidx = vec![0u32; QB * k];
            let mut worst_v = [T::ZERO; QB];
            let mut worst_i = [0u32; QB];
            let mut xn = [T::ZERO; QB];

            for b in lo..hi {
                let q0 = b * QB;
                let active = (n_query - q0).min(QB);

                stage_tile(xq, &mut xtile, q0, active, d);
                if MID == MID_COSINE {
                    for (a, slot) in xn.iter_mut().enumerate() {
                        *slot = if a < active { xnorm[q0 + a] } else { T::ZERO };
                    }
                }
                prefill(&mut lval, &mut lidx, &mut worst_v, &mut worst_i, active, k);

                dispatch_scan_block::<T, MID, PI>(
                    xt,
                    &xtile,
                    &xn,
                    &ynorm,
                    n_train,
                    d,
                    k,
                    p,
                    &mut lval,
                    &mut lidx,
                    &mut worst_v,
                    &mut worst_i,
                );

                emit::<T, MID>(&lval, &lidx, out_val, out_idx, q0, active, k, p);
            }
        };
        pool.run(&pass);
    }

    (out_val, out_idx)
}

/// Stage one block's query rows into the TRANSPOSED tile `xt[c][a]`.
///
/// Lanes beyond `active` (a trailing partial block) stage zeros; [`prefill`]
/// disables them so they never admit, and [`emit`] never reads them.
#[inline]
pub(super) fn stage_tile<T: HostFloat>(xq: &[T], xtile: &mut [[T; QB]], q0: usize, active: usize, d: usize) {
    for (c, slot) in xtile.iter_mut().enumerate().take(d) {
        for (a, lane) in slot.iter_mut().enumerate() {
            *lane = if a < active {
                xq[(q0 + a) * d + c]
            } else {
                T::ZERO
            };
        }
    }
}

/// Sentinel-prefill the block's `k`-lists, and DISABLE the inactive lanes.
///
/// Every active lane's list is filled with `(+∞, u32::MAX)`, the pair that sorts
/// after every real candidate under the `(value, index)` order — so admission
/// rejects with a single compare instead of tracking a fill count, and the `k`
/// slots are all displaced by real candidates before the scan ends (`k <=
/// n_train` is validated by the caller). An inactive lane gets `−∞`, which no
/// distance can beat, so a trailing block's dummy lanes are screened out on every
/// training row instead of maintaining a top-k the emit then discards.
///
/// This is [`prefill_lists`](mlrs_kernels::knn)'s rule with the list stride `k`
/// instead of the kernel's comptime 16.
#[inline]
fn prefill<T: HostFloat>(
    lval: &mut [T],
    lidx: &mut [u32],
    worst_v: &mut [T; QB],
    worst_i: &mut [u32; QB],
    active: usize,
    k: usize,
) {
    let inf = T::from_f64(f64::INFINITY);
    for a in 0..QB {
        for j in 0..k {
            lval[a * k + j] = inf;
            lidx[a * k + j] = u32::MAX;
        }
        worst_v[a] = if a < active {
            inf
        } else {
            T::from_f64(f64::NEG_INFINITY)
        };
        worst_i[a] = u32::MAX;
    }
}

/// Run one block's scan through the WIDEST vector unit this machine actually
/// has, instead of the one the crate was compiled for.
///
/// The whole rationale — why the crate is compiled for the x86-64 baseline while
/// `cubecl-cpu` JITs for the host, why widening the vectors cannot change a
/// result, and why this is written as an explicit TWIN rather than a closure
/// helper — lives on [`host_simd`](super::host_simd), which is where it was
/// first measured (1.6-1.9x on every metric here).
/// `knn_host_test::avx2_and_baseline_agree_bitwise` is the bitwise-identity gate.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dispatch_scan_block<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    k: usize,
    p: f64,
    lval: &mut [T],
    lidx: &mut [u32],
    worst_v: &mut [T; QB],
    worst_i: &mut [u32; QB],
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: guarded by the runtime detection this branch tests, and the
        // body is the ordinary `scan_block` — nothing in it is unsafe on its own.
        unsafe {
            scan_block_avx2::<T, MID, PI>(
                xt, xtile, xn, ynorm, n_train, d, k, p, lval, lidx, worst_v, worst_i,
            );
        }
        return;
    }
    scan_block::<T, MID, PI>(
        xt, xtile, xn, ynorm, n_train, d, k, p, lval, lidx, worst_v, worst_i,
    );
}

/// [`scan_block`] compiled for AVX2 + FMA — see [`dispatch_scan_block`].
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`
/// ([`avx2_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn scan_block_avx2<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    k: usize,
    p: f64,
    lval: &mut [T],
    lidx: &mut [u32],
    worst_v: &mut [T; QB],
    worst_i: &mut [u32; QB],
) {
    scan_block::<T, MID, PI>(
        xt, xtile, xn, ynorm, n_train, d, k, p, lval, lidx, worst_v, worst_i,
    );
}

/// One block's full training scan: every training row, once.
///
/// Split out of [`scan`] so the hot nest is a self-contained function the
/// optimizer sees with all of its operands as locals, and so
/// [`dispatch_scan_block`] can instantiate it a second time under a wider target
/// feature set.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scan_block<T: HostFloat, const MID: u32, const PI: u32>(
    xt: &[T],
    xtile: &[[T; QB]],
    xn: &[T; QB],
    ynorm: &[T],
    n_train: usize,
    d: usize,
    k: usize,
    p: f64,
    lval: &mut [T],
    lidx: &mut [u32],
    worst_v: &mut [T; QB],
    worst_i: &mut [u32; QB],
) {
    let pe = T::from_f64(p);

    for t in 0..n_train {
        let yrow = &xt[t * d..t * d + d];
        let yn = if MID == MID_COSINE { ynorm[t] } else { T::ZERO };
        let acc = lane_distances::<T, MID, PI>(xtile, yrow, xn, yn, pe);

        // Screen the whole block with one pass before touching any list:
        // admission is a REJECT for the overwhelming majority of training rows
        // (a query row admits ~k + O(k·ln n_train) candidates out of n_train),
        // so proving "no lane can admit" cheaply is most of the work. `<=` and
        // not `<`, because an equal value still has the index tie-break to win.
        let mut hit = false;
        for a in 0..QB {
            hit |= acc[a] <= worst_v[a];
        }
        if hit {
            for a in 0..QB {
                insert(lval, lidx, worst_v, worst_i, a, acc[a], t as u32, k);
            }
        }
    }
}

/// One training row's distance to all `QB` staged query rows, under metric
/// `MID` — the vectorizable core BOTH host scans are built on.
///
/// Returns the metric's INTERNAL (pre-boundary-transform) value: squared for
/// Euclidean, `Σ|Δ|^p` for Minkowski, and the true distance for
/// Manhattan/Chebyshev/Cosine. A monotone boundary transform (`sqrt`, `^(1/p)`)
/// is applied by the caller to the values it actually keeps — the `k` selected
/// here, the matches within `radius` in [`radius_host`](super::radius_host) —
/// never to all `n_train` candidates.
///
/// Split out so the radius scan reuses this loop VERBATIM rather than growing a
/// second copy that could drift on a tie, a NaN, or a clamp; `#[inline(always)]`
/// keeps it a straight-line body inside each caller's `#[target_feature]` twin,
/// which is where the width the machine actually has comes from
/// ([`dispatch_scan_block`]).
///
/// `xn`/`yn` are the RECIPROCAL row norms Cosine folds in (`row_inv_norm_host`);
/// every other metric ignores them. `pe` is `p` widened to `T`, read only by the
/// non-integer Minkowski arm.
#[inline(always)]
pub(super) fn lane_distances<T: HostFloat, const MID: u32, const PI: u32>(
    xtile: &[[T; QB]],
    yrow: &[T],
    xn: &[T; QB],
    yn: T,
    pe: T,
) -> [T; QB] {
    // The QB lanes are INDEPENDENT reductions, which is what lets this
    // vectorize — a reduction over the feature axis could not, because Rust
    // will not reassociate float addition. Zipping the tile against the
    // training row walks both without a bounds check per feature.
    let mut acc = [T::ZERO; QB];
    for (xc, &yv) in xtile.iter().zip(yrow.iter()) {
        if MID == MID_EUCLIDEAN {
            for a in 0..QB {
                let diff = xc[a] - yv;
                acc[a] = acc[a] + diff * diff;
            }
        } else if MID == MID_MANHATTAN {
            for a in 0..QB {
                acc[a] = acc[a] + (xc[a] - yv).abs();
            }
        } else if MID == MID_CHEBYSHEV {
            for a in 0..QB {
                let v = (xc[a] - yv).abs();
                // A SELECT, not a conditional store: written as `if v >
                // acc[a] { acc[a] = v }` the lane loop measured 5x slower
                // than the Manhattan one it is otherwise identical to (0.044
                // s against 0.009 s at 20_000 x 2_000 x 16) — LLVM will not
                // vectorize a loop whose store is predicated, and this is the
                // same running max written so it is unconditional.
                acc[a] = if v > acc[a] { v } else { acc[a] };
            }
        } else if MID == MID_MINKOWSKI && PI > 0 {
            for a in 0..QB {
                let ad = (xc[a] - yv).abs();
                // `PI` is a constant, so this is an unrolled multiply chain
                // and the whole lane loop stays vectorizable.
                let mut r = ad;
                for _ in 1..PI {
                    r = r * ad;
                }
                acc[a] = acc[a] + r;
            }
        } else if MID == MID_MINKOWSKI {
            for a in 0..QB {
                acc[a] = acc[a] + (xc[a] - yv).abs().powf(pe);
            }
        } else {
            // Cosine accumulates the plain dot product; the norms are folded
            // in below, once per training row instead of once per feature.
            for a in 0..QB {
                acc[a] = acc[a] + xc[a] * yv;
            }
        }
    }

    if MID == MID_COSINE {
        // `1 − dot·(1/‖x‖)·(1/‖y‖)`, then the `[0, 2]` clamp sklearn's
        // `cosine_distances` applies. Both reciprocals are precomputed
        // (`row_inv_norm_host`) and the training row's is hoisted out of the
        // lane loop, so this epilogue is two multiplies and two selects per
        // lane — no root and no divide anywhere in the scan.
        let one = T::ONE;
        let two = T::lit(2.0);
        for a in 0..QB {
            let mut dv = one - acc[a] * xn[a] * yn;
            dv = if dv < T::ZERO { T::ZERO } else { dv };
            dv = if dv > two { two } else { dv };
            acc[a] = dv;
        }
    }
    acc
}

/// THE admission rule — [`insert_lane`](mlrs_kernels::knn)'s, with the list
/// stride `k`.
///
/// Folds candidate `(v, t)` into lane `a`'s sorted list or rejects it, under the
/// `(value, index)` total order that gives ties to the lowest training index.
/// Keeping the rule identical to the kernels' is what makes the two arms agree
/// on a tie rather than agreeing only where the data is generic.
#[allow(clippy::too_many_arguments)]
#[inline]
fn insert<T: HostFloat>(
    lval: &mut [T],
    lidx: &mut [u32],
    worst_v: &mut [T; QB],
    worst_i: &mut [u32; QB],
    a: usize,
    v: T,
    t: u32,
    k: usize,
) {
    let w = worst_v[a];
    // NaN admits nothing: both compares are false, exactly as in the kernel.
    let admit = v < w || (v == w && t < worst_i[a]);
    if !admit {
        return;
    }
    let base = a * k;
    let mut cav = v;
    let mut cai = t;
    for j in 0..k {
        let jv = lval[base + j];
        let ji = lidx[base + j];
        // The incoming pair takes slot `j`, and the pair that was there becomes
        // the incoming one — an insertion sort over an already-sorted list.
        if cav < jv || (cav == jv && cai < ji) {
            lval[base + j] = cav;
            lidx[base + j] = cai;
            cav = jv;
            cai = ji;
        }
    }
    worst_v[a] = lval[base + k - 1];
    worst_i[a] = lidx[base + k - 1];
}

/// Write one block's finished lists out, applying the metric's BOUNDARY
/// transform and clamping any surviving sentinel index.
///
/// The root is applied here — to the `k` selected values — and not to every one
/// of the `n_train` candidates, because a root is monotone and so cannot change
/// which pairs are selected or their order. That is the same deferral
/// [`top_k`](super::topk::top_k) performs for the Euclidean device path.
///
/// A query row whose distances are all NaN admits nothing and still holds its
/// `(+∞, u32::MAX)` sentinels; the index is clamped to 0 so nothing out of range
/// reaches a consumer, and the paired `+∞` is what tells the caller the row had
/// no finite neighbour ([`emit_lists`](mlrs_kernels::knn) does the same).
#[allow(clippy::too_many_arguments)]
#[inline]
fn emit<T: HostFloat, const MID: u32>(
    lval: &[T],
    lidx: &[u32],
    out_val: &mut [T],
    out_idx: &mut [u32],
    q0: usize,
    active: usize,
    k: usize,
    p: f64,
) {
    let inv_p = T::from_f64(1.0 / p);
    for a in 0..active {
        for j in 0..k {
            let v = lval[a * k + j];
            out_val[(q0 + a) * k + j] = if MID == MID_EUCLIDEAN {
                v.sqrt()
            } else if MID == MID_MINKOWSKI {
                v.powf(inv_p)
            } else {
                v
            };
            let ix = lidx[a * k + j];
            out_idx[(q0 + a) * k + j] = if ix == u32::MAX { 0 } else { ix };
        }
    }
}
