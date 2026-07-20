//! `prims::kmeans` — host orchestration for the KMeans primitives (CLUSTER-01,
//! D-01).
//!
//! The validate-before-launch wrappers for the Lloyd iteration math and the
//! k-means++ default init, composing the Phase-2 distance prim and the new
//! `mlrs_kernels::kmeans` GATHER kernels:
//!
//! - [`lloyd_update`] — recompute each centroid as the MEAN of its assigned
//!   samples (`centroid_sumcount` device gather → host divide), with sklearn's
//!   empty-cluster RELOCATION (an empty cluster is moved to the point farthest
//!   from its current center, never a divide-by-zero NaN — T-05-03-02).
//! - [`inertia`] — `Σ_i ‖X_i − centers[labels_i]‖²` (`inertia_rows` device
//!   gather → host sum), the squared, no-sqrt sum-of-distances-to-assigned-center
//!   (Pitfall 8 / D-08).
//! - [`kmeanspp_sample`] — the k-means++ D²-weighted default init: a HOST-side
//!   seeded PRNG draws each next center with probability ∝ its squared distance
//!   to the nearest already-chosen center (D-09a/c). The D² weights are computed
//!   on-device (via the distance prim) and read back ONCE PER CENTER at init
//!   only (NOT the Lloyd hot loop — D-09c); the RNG itself is host-side and
//!   documented-seeded (never `OsRng` — ASVS V6 / T-05-03-03), so the sampler is
//!   seed-reproducible across runs and backends.
//!
//! ## Validate before any unsafe launch (T-05-03-01 / ASVS V5)
//! Every entry point validates its geometry (`n * d == x.len()`, `1 <= k <= n`,
//! label/center shapes) and returns a [`PrimError::ShapeMismatch`] BEFORE any
//! `unsafe` device launch (the `distance.rs` / `topk.rs` precedent — `PrimError`
//! has no dedicated `InvalidK` variant, so a bad `k` surfaces as a `"k"`
//! ShapeMismatch).
//!
//! ## Assignment reuses the existing prims (do NOT rebuild)
//! Nearest-centroid assignment is `argmin_rows` over the `distance(X, centers,
//! sqrt=false)` matrix — the Phase-2 distance prim + `prims::reduce::argmin_rows`
//! — so this module never re-implements pairwise distance or argmin.
//!
//! Tests live in `crates/mlrs-backend/tests/{kmeanspp,lloyd}_test.rs`
//! (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64};
use mlrs_core::PrimError;
use mlrs_kernels::dist_combine_clamp;
use mlrs_kernels::kmeans::{
    argmin_dist_rows, block_sum_f, centroid_reduce_partials, centroid_sumcount,
    centroid_sumcount_blocked, centroid_sumcount_shared, col_sqdiff_blocked, col_sum_blocked,
    count_blocked, count_reduce, dist_direct_2d, dist_direct_2d_c4, gather_rows_idx, inertia_rows,
    kmeanspp_mind2, labels_diff_blocked, onehot_from_labels, row_sqnorm,
};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::prims::rng::SplitMix64;
use crate::runtime::ActiveRuntime;

/// Assignment stage-1 path choice: GEMM expansion (`dist = max(‖x‖² + ‖c‖² −
/// 2XCᵀ, 0)` on the tiled matmul — the cuML shape) vs the direct per-`(i, c)`
/// gather. Measured on the T4/wgpu perf ladders:
///
/// - cpu: ALWAYS direct — cubek-matmul uses `SharedMemory`, whose cubecl-cpu
///   barrier emulation is the PyO3-embedded livelock landmine (see the t-SNE
///   `row_reduce(Shared)` memory), and the MLIR gate is the primary
///   correctness environment.
/// - wgpu: direct — the per-call matmul cost made the GEMM path 10–50×
///   SLOWER than the direct kernels on the wgpu ladder.
/// - cuda: total-time A/B on the T4 showed the GEMM path SLOWER there too
///   once lap attribution was corrected (the `labels_changed` readback does
///   not drain cubek-matmul's work, so its cost had been mis-attributed to
///   the next phase's lap) — e.g. 100k×64×32 direct 0.37s vs GEMM 1.07s.
///
/// Default is therefore DIRECT everywhere; `KM_GEMM=1` opts back into the
/// GEMM path for measurement (never on cpu, the landmine).
fn use_gemm_assign(d: usize, k: usize) -> bool {
    let _ = (d, k);
    #[cfg(feature = "cpu")]
    {
        false
    }
    #[cfg(not(feature = "cpu"))]
    {
        std::env::var("KM_GEMM").is_ok()
    }
}

/// Centroid-sums stage-1 path choice. The GEMM formulation (`sums = onehotᵀX`)
/// measured CATASTROPHICALLY slow on every backend (a skinny `k×d`-output
/// reduction over the huge `n` axis — no split-K in cubek-matmul), so it is
/// reachable only via `KM_GEMM_SUMS=1`. GPU backends use the deterministic
/// SHARED-MEMORY kernel when the whole `k × d` accumulator fits its fixed
/// 4096-slot tile (O(n·d) work instead of the gather's O(n·k·d)); the
/// row-blocked GATHER covers the rest and the cpu backend always (MLIR rejects
/// `SharedMemory`; `KM_SUMS_GATHER=1` forces it everywhere for A/B).
fn use_shared_sums(kd: usize) -> bool {
    #[cfg(feature = "cpu")]
    {
        let _ = kd;
        false
    }
    #[cfg(not(feature = "cpu"))]
    {
        if std::env::var("KM_SUMS_GATHER").is_ok() {
            return false;
        }
        kd <= 4096
    }
}

/// Recompute the `k × d` centroids as the per-label MEAN of the assigned rows of
/// `x` (`n × d`), implementing sklearn's empty-cluster relocation (CLUSTER-01).
///
/// - `x` is the row-major `n × d` sample matrix (device-resident).
/// - `labels` is the length-`n` cluster assignment (`u32`, each in `0..k`).
/// - `(n, d, k)` is the geometry; validated (`n * d == x.len()`,
///   `labels.len() == n`, `1 <= k <= n`) BEFORE any launch (T-05-03-01).
///
/// The device [`centroid_sumcount`] gather produces the per-centroid feature
/// sums + counts; the host divides each sum by its count to form the mean. An
/// EMPTY cluster (`count == 0`) is RELOCATED exactly like sklearn's
/// `_relocate_empty_clusters_dense` (T-05-03-02): the empty clusters take, in
/// order, the GLOBALLY-farthest samples (by `dist_to_assigned[i]` — the squared
/// distance of sample `i` to ITS currently-assigned center, supplied by the
/// caller). Relocating sample `i` to empty cluster `c` sets `centers[c] = X[i]`,
/// moves `X[i]` from its donor cluster's running sum to `c`'s, increments `c`'s
/// count and DECREMENTS the donor's. The mean is therefore never a
/// divide-by-zero NaN, and the result matches sklearn (not the old "farthest from
/// the nearest non-empty center" approximation, which had no sklearn analogue).
///
/// `dist_to_assigned` is the length-`n` per-sample squared distance to the
/// assigned center under `labels` (compute it with [`inertia_rows_host`] against
/// the SAME centers that produced `labels`). It is only consulted when an empty
/// cluster exists; pass a correct slice regardless so the relocation is exact.
///
/// Returns the `k × d` centroids as a device-resident [`DeviceArray`] (D-05).
/// Generic over `F` (`f32` / `f64`); the f64 path is caller-gated by
/// `skip_f64_with_log`.
pub fn lloyd_update<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    labels: &[u32],
    dist_to_assigned: &[f64],
    n: usize,
    d: usize,
    k: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), n, d, k)?;
    if labels.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len(),
        });
    }
    if dist_to_assigned.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "dist_to_assigned",
            rows: n,
            cols: 1,
            len: dist_to_assigned.len(),
        });
    }
    // Defensive: every label must address a real centroid (0..k) so the device
    // gather never reads/writes out of the k×d sums buffer (mitigates T-05-03-01
    // — a tampered label is a recoverable typed error, not an OOB device write).
    for (i, &l) in labels.iter().enumerate() {
        if (l as usize) >= k {
            return Err(PrimError::ShapeMismatch {
                operand: "labels",
                rows: i,
                cols: l as usize,
                len: k,
            });
        }
    }

    // --- Device gather: per-centroid feature sums (k × d) + counts (k). ---
    let labels_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, labels);
    let sums_len = k * d;
    let sums_handle = pool.acquire(sums_len * size_of::<F>());
    let counts_handle = pool.acquire(k * size_of::<u32>());

    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(sums_len);

    // SAFETY: lengths are the validated element counts; the kernel bounds-checks
    // `tid < k*d` and only gathers `n` rows (mitigates T-05-03-01).
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let lab_arg = unsafe { ArrayArg::from_raw_parts(labels_dev.handle().clone(), n) };
    let sums_arg = unsafe { ArrayArg::from_raw_parts(sums_handle.clone(), sums_len) };
    let cnt_arg = unsafe { ArrayArg::from_raw_parts(counts_handle.clone(), k) };

    centroid_sumcount::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        x_arg,
        lab_arg,
        sums_arg,
        cnt_arg,
        n as u32,
        d as u32,
        k as u32,
    );

    let sums_dev = DeviceArray::<ActiveRuntime, F>::from_raw(sums_handle, sums_len);
    let counts_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(counts_handle, k);
    let sums_host: Vec<F> = sums_dev.to_host(pool);
    let counts_host: Vec<u32> = counts_dev.to_host(pool);
    sums_dev.release_into(pool);
    counts_dev.release_into(pool);

    // --- Host finalize: sklearn `_relocate_empty_clusters_dense` BEFORE the
    //     divide, then divide each (possibly relocation-adjusted) sum by its count
    //     → mean. The relocation mutates the running sums + counts so the donor
    //     cluster correctly loses the moved point's contribution (T-05-03-02). ---
    let x_host: Vec<F> = x.to_host(pool);

    // Promote the device sums to f64 + carry counts as i64 so the relocation can
    // add/subtract a moved point's features and decrement a donor count exactly.
    let mut sums_f64: Vec<f64> = sums_host.iter().map(|&s| host_to_f64(s)).collect();
    let mut counts_i64: Vec<i64> = counts_host.iter().map(|&c| c as i64).collect();

    relocate_empty_clusters::<F>(
        &mut sums_f64,
        &mut counts_i64,
        &x_host,
        labels,
        dist_to_assigned,
        n,
        d,
        k,
    )?;

    let mut centers: Vec<F> = vec![F::from_int(0); sums_len];
    for c in 0..k {
        // After relocation every cluster has count >= 1: each empty cluster
        // received exactly one point from a donor that retained >= 1 member, so
        // no count can be 0 or negative here. The `> 0` check is a defensive
        // invariant assertion; the relocation loop above is what guarantees it.
        debug_assert!(
            counts_i64[c] > 0,
            "post-relocation cluster {c} has non-positive count {}",
            counts_i64[c]
        );
        if counts_i64[c] > 0 {
            let inv = 1.0_f64 / counts_i64[c] as f64;
            for j in 0..d {
                centers[c * d + j] = f64_to_host::<F>(sums_f64[c * d + j] * inv);
            }
        }
    }

    labels_dev.release_into(pool);
    Ok(DeviceArray::from_host(pool, &centers))
}

/// Compute the KMeans inertia `Σ_i ‖X_i − centers[labels_i]‖²` over `x`
/// (`n × d`) and `centers` (`k × d`) under the assignment `labels` (CLUSTER-01).
///
/// Squared, NO sqrt (Pitfall 8 / D-08). Geometry validated BEFORE launch
/// (`n * d == x.len()`, `labels.len() == n`, each label `< k`). The device
/// [`inertia_rows`] gather produces the `n` per-row squared distances; the host
/// sums them to the scalar inertia. Returns the scalar `F`.
pub fn inertia<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    centers: &DeviceArray<ActiveRuntime, F>,
    labels: &[u32],
    n: usize,
    d: usize,
) -> Result<F, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // n*d == x.len(); centers is k×d for some k = centers.len()/d.
    if n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    if d == 0 || centers.len() % d != 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "centers",
            rows: 0,
            cols: d,
            len: centers.len(),
        });
    }
    let k = centers.len() / d;
    // WR-06: reject an empty `centers` buffer (k == 0). `centers.len() % d == 0`
    // is satisfied by `len == 0`, so without this guard the function would
    // proceed with k == 0 — an inconsistent surface vs `validate_geometry`
    // (which enforces `1 <= k`).
    if k == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "centers",
            rows: 0,
            cols: d,
            len: centers.len(),
        });
    }
    if labels.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len(),
        });
    }
    for (i, &l) in labels.iter().enumerate() {
        if (l as usize) >= k {
            return Err(PrimError::ShapeMismatch {
                operand: "labels",
                rows: i,
                cols: l as usize,
                len: k,
            });
        }
    }
    // WR-03: n, d are cast to u32 for the launch.
    guard_u32("n", n)?;
    guard_u32("d", d)?;

    let labels_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, labels);
    let out_handle = pool.acquire(n * size_of::<F>());

    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);

    // SAFETY: validated element counts; the kernel bounds-checks `i < n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(centers.handle().clone(), centers.len()) };
    let lab_arg = unsafe { ArrayArg::from_raw_parts(labels_dev.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };

    inertia_rows::launch::<F, ActiveRuntime>(
        &client, count, dim, x_arg, c_arg, lab_arg, out_arg, n as u32, d as u32,
    );

    let out_dev = DeviceArray::<ActiveRuntime, F>::from_raw(out_handle, n);
    let parts: Vec<F> = out_dev.to_host(pool);
    out_dev.release_into(pool);
    labels_dev.release_into(pool);

    // Host sum of the n per-row squared distances (small-k/n finalize, RESEARCH
    // Open Q1: keep the device work on the n-heavy per-row distance, finalize on
    // the host). f64 accumulate for stability even in the f32 case.
    let total: f64 = parts.iter().map(|&p| host_to_f64(p)).sum();
    Ok(f64_to_host::<F>(total))
}

/// k-means++ D²-weighted default init (D-09a/c): draw `k` distinct center INDICES
/// from the `n × d` sample matrix `x`, the first uniformly and each subsequent
/// with probability ∝ its squared distance to the nearest already-chosen center.
///
/// - `(n, d)` is the sample geometry; `k` centers are drawn; validated
///   (`n * d == x.len()`, `1 <= k <= n`) BEFORE any launch (T-05-03-01).
/// - `seed` seeds a HOST-side documented PRNG ([`SplitMix64`]) — never `OsRng`
///   (ASVS V6 / T-05-03-03), so the same `seed` yields the SAME indices across
///   runs and backends for THIS implementation (an mlrs-specific, deterministic
///   sampler — NOT bit-identical to sklearn's MT19937 `randint`/weighted draw).
///   The deterministic oracle injects a fixed init via [`KMeans::with_init`], so
///   sklearn-parity does not depend on this sampler reproducing sklearn's stream;
///   it only needs to be unbiased + reproducible for a given seed.
///
/// The D² weights are computed ON-DEVICE via the [`distance`] prim (squared, no
/// sqrt) and read back to the host ONCE PER CENTER — at INIT ONLY, not the Lloyd
/// hot loop (D-09c). There is NO device-side RNG (backend-divergent — RESEARCH
/// Anti-Patterns); the weighted draw is a host cumulative-weight scan.
///
/// Returns the `k` chosen sample indices (host `Vec<usize>`); the caller gathers
/// the corresponding rows of `x` to form the `k × d` init centers.
pub fn kmeanspp_sample<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<usize>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), n, d, k)?;

    let mut rng = SplitMix64::new(seed);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    // WR-05: O(1) membership test (avoids the O(k·n) repeated `chosen.contains`)
    // AND lets the degenerate fallback return a typed error instead of `expect`.
    let mut is_chosen: Vec<bool> = vec![false; n];

    // Center 0: UNBIASED uniform over 0..n (host PRNG, not device). A plain
    // `next_u64() % n` is biased for `n` not a power of two (CR-02); use
    // rejection sampling so every index is equally likely.
    let first = rng.next_below(n as u64) as usize;
    chosen.push(first);
    is_chosen[first] = true;

    // Running nearest-center squared distance per sample, DEVICE-resident: the
    // fused [`kmeanspp_mind2`] kernel computes each row's D² to the new center
    // (read straight from `x` by index — no center upload, no `x` host copy)
    // and folds the running min in place. One launch + one n-float readback
    // per drawn center (INIT-only, D-09c) — the host copy feeds the weighted
    // draw below.
    let mind2_dev =
        DeviceArray::<ActiveRuntime, F>::from_raw(pool.acquire(n * size_of::<F>()), n);
    let mut min_d2: Vec<f64> = mind2_update_readback::<F>(pool, x, &mind2_dev, n, d, first, true)?;

    while chosen.len() < k {
        // Weighted draw ∝ min_d2 (D²). Host cumulative-weight scan over the
        // device-computed weights (RNG is host-side, D-09c). Exclude already
        // chosen indices (their D² is 0, so they have zero weight anyway, but a
        // tie-at-zero must never re-pick).
        let total: f64 = min_d2.iter().sum();
        let next = if total <= 0.0 {
            // All remaining samples coincide with a chosen center (degenerate /
            // duplicate data): fall back to the first not-yet-chosen index so we
            // still return k DISTINCT centers. WR-05: the `chosen.len() < k <= n`
            // invariant normally guarantees an unused index, but a future caller
            // (e.g. k == n on all-duplicate data where every index is already
            // chosen) could violate it — return a typed error, never panic across
            // the boundary.
            match (0..n).find(|&i| !is_chosen[i]) {
                Some(i) => i,
                None => {
                    return Err(PrimError::ShapeMismatch {
                        operand: "k",
                        rows: 1,
                        cols: k,
                        len: n,
                    });
                }
            }
        } else {
            let target = rng.next_f64() * total;
            let mut acc = 0.0_f64;
            let mut pick = n - 1;
            for (i, &w) in min_d2.iter().enumerate() {
                acc += w;
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            // CR-02: under f64 rounding the accumulated `acc` can fall a few ULP
            // short of `total`, so when `target` rounds to ~`total` the scan never
            // triggers `acc >= target` and falls through to the `pick = n - 1`
            // initializer — selecting the LAST sample regardless of its weight. If
            // the picked sample has non-positive weight, fall back to the last
            // POSITIVE-weight index (a real D²-weighted draw, not the fall-through).
            if min_d2[pick] <= 0.0 {
                pick = min_d2.iter().rposition(|&w| w > 0.0).unwrap_or(pick);
            }
            // Guard against re-picking an already-chosen index (possible only via
            // a zero-weight rounding edge): walk forward to the next unused one.
            if is_chosen[pick] {
                (0..n).find(|&i| !is_chosen[i]).unwrap_or(pick)
            } else {
                pick
            }
        };
        chosen.push(next);
        is_chosen[next] = true;

        // Update the running min-D² on the DEVICE (the fused kernel folds
        // `min(old, d²-to-new-center)` in place — the min of two floats is
        // exact, so folding in F instead of f64 loses nothing) and read the
        // folded buffer back for the next draw (D-09c).
        min_d2 = mind2_update_readback::<F>(pool, x, &mind2_dev, n, d, next, false)?;
    }

    mind2_dev.release_into(pool);
    Ok(chosen)
}

/// Per-sample squared distance to the ASSIGNED center
/// (`‖X_i − centers[labels_i]‖²` for every `i`), as a host `Vec<f64>` of length
/// `n` (CLUSTER-01, the sklearn `_relocate_empty_clusters_dense` input).
///
/// This is the same device [`inertia_rows`] gather that [`inertia`] consumes, but
/// the per-row distances are returned to the host UNSUMMED so the estimator's
/// Lloyd loop can rank samples by their distance-to-assigned-center for sklearn's
/// empty-cluster relocation. Geometry validated BEFORE launch exactly like
/// [`inertia`].
pub fn inertia_rows_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    centers: &DeviceArray<ActiveRuntime, F>,
    labels: &[u32],
    n: usize,
    d: usize,
) -> Result<Vec<f64>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    if d == 0 || centers.len() % d != 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "centers",
            rows: 0,
            cols: d,
            len: centers.len(),
        });
    }
    let k = centers.len() / d;
    // WR-06: reject an empty `centers` buffer (k == 0); `len == 0` passes the
    // `centers.len() % d == 0` check above, so guard explicitly for parity with
    // `validate_geometry`'s `1 <= k`.
    if k == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "centers",
            rows: 0,
            cols: d,
            len: centers.len(),
        });
    }
    if labels.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len(),
        });
    }
    for (i, &l) in labels.iter().enumerate() {
        if (l as usize) >= k {
            return Err(PrimError::ShapeMismatch {
                operand: "labels",
                rows: i,
                cols: l as usize,
                len: k,
            });
        }
    }
    // WR-03: n, d are cast to u32 for the launch.
    guard_u32("n", n)?;
    guard_u32("d", d)?;

    let labels_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, labels);
    let out_handle = pool.acquire(n * size_of::<F>());

    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);

    // SAFETY: validated element counts; the kernel bounds-checks `i < n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(centers.handle().clone(), centers.len()) };
    let lab_arg = unsafe { ArrayArg::from_raw_parts(labels_dev.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };

    inertia_rows::launch::<F, ActiveRuntime>(
        &client, count, dim, x_arg, c_arg, lab_arg, out_arg, n as u32, d as u32,
    );

    let out_dev = DeviceArray::<ActiveRuntime, F>::from_raw(out_handle, n);
    let parts: Vec<F> = out_dev.to_host(pool);
    out_dev.release_into(pool);
    labels_dev.release_into(pool);

    Ok(parts.iter().map(|&p| host_to_f64(p)).collect())
}

/// Launch the fused [`kmeanspp_mind2`] running-min-D² update against the
/// sample row `center_idx` (`first` seeds the buffer on the first center) and
/// read the folded buffer back as `f64` weights for the host draw (D-09c).
fn mind2_update_readback<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    mind2: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    center_idx: usize,
    first: bool,
) -> Result<Vec<f64>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: lengths are the carried element counts; the kernel bounds-checks
    // `i < n` and `center_idx < n` is guaranteed by the caller's draw domain.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let m_arg = unsafe { ArrayArg::from_raw_parts(mind2.handle().clone(), n) };
    kmeanspp_mind2::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        x_arg,
        m_arg,
        center_idx as u32,
        if first { 1u32 } else { 0u32 },
        n as u32,
        d as u32,
    );
    Ok(mind2.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
}

// ===========================================================================
// DEVICE-RESIDENT Lloyd-loop prims (the launch-only hot path)
//
// The Lloyd estimator loop composes these so the per-iteration host traffic is
// a few KB (per-centroid sums/counts + per-block changed counts) instead of
// the O(n·d) `x` readbacks + per-row argmin launches of the original path —
// the same "count synchronizations, not FLOPs" treatment that fixed sgd_solve
// and the Random Forest builder (see the GPU-perf memory).
// ===========================================================================

/// FUSED nearest-center assignment into CALLER-OWNED device buffers: writes
/// `labels[i] = argmin_c ‖x_i − centers_c‖²` (`u32`, lowest-index tie-break
/// D-02) and `dist[i]` = the winning squared distance (the per-row inertia
/// term). No readback — labels stay device-resident for the update gather.
///
/// Two launch-only stage-1 variants fill the `n × k` staging matrix (see
/// [`use_gemm_path`]): the direct per-`(i, c)` GATHER, or the GEMM expansion
/// `max(‖x_i‖² + ‖c_j‖² − 2·XCᵀ, 0)` (tiled matmul cross term + the validated
/// `dist_combine_clamp` combine — the cuML shape). The GEMM variant needs the
/// per-fit `xnorm` (`‖x_i‖²`, from [`row_sqnorms`]); pass `None` to force the
/// direct kernels (e.g. when no norms are cached).
#[allow(clippy::too_many_arguments)]
pub fn assign_min<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    centers: &DeviceArray<ActiveRuntime, F>,
    labels: &DeviceArray<ActiveRuntime, u32>,
    dist: &DeviceArray<ActiveRuntime, F>,
    xnorm: Option<&DeviceArray<ActiveRuntime, F>>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<(), PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), n, d, k)?;
    if centers.len() != k * d {
        return Err(PrimError::ShapeMismatch {
            operand: "centers",
            rows: k,
            cols: d,
            len: centers.len(),
        });
    }
    if labels.len() != n || dist.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len().min(dist.len()),
        });
    }
    if let Some(xn) = xnorm {
        if xn.len() != n {
            return Err(PrimError::ShapeMismatch {
                operand: "xnorm",
                rows: n,
                cols: 1,
                len: xn.len(),
            });
        }
    }

    guard_u32("n*k", n * k)?;
    let client = pool.client().clone();
    let dmat_len = n * k;

    // Stage 1 into the n × k staging matrix (a pool acquisition — the
    // free-list hands the same buffer back every iteration). Deliberately
    // split from the argmin (single short loops; a fused per-row nested k × d
    // loop compiled pathologically on wgpu).
    let dmat = match xnorm {
        Some(xn) if use_gemm_assign(d, k) => {
            // ‖c_j‖² per center (tiny k-unit launch, once per iteration).
            let cnorm = pool.acquire(k * size_of::<F>());
            // SAFETY: validated element counts; kernels bounds-check unit ids.
            let c_arg = unsafe { ArrayArg::from_raw_parts(centers.handle().clone(), centers.len()) };
            let cn_arg = unsafe { ArrayArg::from_raw_parts(cnorm.clone(), k) };
            let (cc, cd) = launch_dims_1d(k);
            row_sqnorm::launch::<F, ActiveRuntime>(&client, cc, cd, c_arg, cn_arg, k as u32, d as u32);

            // Cross term XCᵀ (n × k) on the tiled matmul.
            let dmat_arr =
                DeviceArray::<ActiveRuntime, F>::from_raw(pool.acquire(dmat_len * size_of::<F>()), dmat_len);
            let xy = gemm::<F>(pool, x, (n, d), centers, (d, k), false, true, Some(dmat_arr))?;

            // Combine + clamp IN PLACE over the cross term (element-wise same
            // index, the sqrt_elem in-place precedent): max(xn + cn − 2xy, 0).
            let xy_arg = unsafe { ArrayArg::from_raw_parts(xy.handle().clone(), dmat_len) };
            let out_arg = unsafe { ArrayArg::from_raw_parts(xy.handle().clone(), dmat_len) };
            let xn_arg = unsafe { ArrayArg::from_raw_parts(xn.handle().clone(), n) };
            let cn_arg2 = unsafe { ArrayArg::from_raw_parts(cnorm.clone(), k) };
            let (c2, d2) = launch_dims_2d(n, k);
            dist_combine_clamp::launch::<F, ActiveRuntime>(
                &client, c2, d2, xy_arg, xn_arg, cn_arg2, out_arg, n as u32, k as u32,
            );
            pool.release(cnorm, k * size_of::<F>());
            xy
        }
        _ => {
            let dmat = pool.acquire(dmat_len * size_of::<F>());
            // SAFETY: validated element counts; kernels bounds-check unit ids.
            let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
            let c_arg = unsafe { ArrayArg::from_raw_parts(centers.handle().clone(), centers.len()) };
            let dm_arg = unsafe { ArrayArg::from_raw_parts(dmat.clone(), dmat_len) };
            // One unit per (row, 4-center chunk) — the c4 kernel loads each
            // x[i, j] once per chunk instead of once per center.
            // KM_ASSIGN_C1=1 switches to the plain per-(row, center) kernel
            // for A/B measurement.
            if std::env::var("KM_ASSIGN_C1").is_ok() {
                let (count1, dim1) = launch_dims_1d(dmat_len);
                dist_direct_2d::launch::<F, ActiveRuntime>(
                    &client, count1, dim1, x_arg, c_arg, dm_arg, n as u32, d as u32, k as u32,
                );
            } else {
                let (count1, dim1) = launch_dims_1d(n * k.div_ceil(4));
                dist_direct_2d_c4::launch::<F, ActiveRuntime>(
                    &client, count1, dim1, x_arg, c_arg, dm_arg, n as u32, d as u32, k as u32,
                );
            }
            DeviceArray::from_raw(dmat, dmat_len)
        }
    };

    // Stage 2: per-row argmin over the staging matrix → labels + min distance.
    // SAFETY: validated element counts; the kernel bounds-checks `i < n`.
    let dm_arg2 = unsafe { ArrayArg::from_raw_parts(dmat.handle().clone(), dmat_len) };
    let l_arg = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let d_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), n) };
    let (count2, dim2) = launch_dims_1d(n);
    argmin_dist_rows::launch::<F, ActiveRuntime>(
        &client, count2, dim2, dm_arg2, l_arg, d_arg, n as u32, k as u32,
    );

    dmat.release_into(pool);
    Ok(())
}

/// Recompute the per-row squared distance to the ASSIGNED center into `dist`
/// with the DIRECT [`inertia_rows`] gather, from DEVICE-resident labels (no
/// upload, no readback). Called once at the end of fit so the stored
/// `inertia_` keeps direct-form accuracy even when the loop's staging
/// distances came from the GEMM expansion (whose f32 cancellation noise is
/// fine for argmin ranking but exceeds the 1e-5 oracle tolerance when SUMMED).
pub fn inertia_rows_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    centers: &DeviceArray<ActiveRuntime, F>,
    labels: &DeviceArray<ActiveRuntime, u32>,
    dist: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<(), PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    if labels.len() != n || dist.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len().min(dist.len()),
        });
    }
    guard_u32("n", n)?;
    guard_u32("d", d)?;
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: validated element counts; the kernel bounds-checks `i < n`, and
    // the labels were produced by the assign kernels (each `< k`).
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(centers.handle().clone(), centers.len()) };
    let l_arg = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let d_arg = unsafe { ArrayArg::from_raw_parts(dist.handle().clone(), n) };
    inertia_rows::launch::<F, ActiveRuntime>(
        &client, count, dim, x_arg, c_arg, l_arg, d_arg, n as u32, d as u32,
    );
    Ok(())
}

/// Per-row squared L2 norms `‖x_i‖²` (`n`-vector, device-resident) via the
/// direct [`row_sqnorm`] kernel — computed ONCE per fit for the GEMM
/// assignment path (never `row_reduce(Shared)`, the PyO3 landmine).
pub fn row_sqnorms<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    guard_u32("n", n)?;
    guard_u32("d", d)?;
    let out = pool.acquire(n * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: validated element counts; the kernel bounds-checks `i < n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), n) };
    row_sqnorm::launch::<F, ActiveRuntime>(&client, count, dim, x_arg, o_arg, n as u32, d as u32);
    Ok(DeviceArray::from_raw(out, n))
}

/// Per-centroid feature sums + counts from DEVICE-resident labels, via the
/// row-blocked two-stage gather ([`centroid_sumcount_blocked`] →
/// [`centroid_reduce_partials`]). Returns the small `k × d` sums (as `f64`)
/// and length-`k` counts (as `i64`) on the host — the ONLY per-iteration
/// readback of the Lloyd update (a few KB), ready for the host mean/relocation
/// finalize.
pub fn centroid_sums_dev<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    labels: &DeviceArray<ActiveRuntime, u32>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<(Vec<f64>, Vec<i64>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), n, d, k)?;
    if labels.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len(),
        });
    }

    if std::env::var("KM_GEMM_SUMS").is_ok() {
        return centroid_sums_gemm::<F>(pool, x, labels, n, d, k);
    }
    if use_shared_sums(k * d) {
        return centroid_sums_shared::<F>(pool, x, labels, n, d, k);
    }

    // Row-block layout: enough blocks for occupancy (~n/256, floor 1), capped
    // so the partial buffer stays ≤ ~8M elements even at large k·d.
    let kd = k * d;
    let nb_cap = ((8usize << 20) / kd.max(1)).max(64);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);

    let psums_len = nb * kd;
    let psums = pool.acquire(psums_len * size_of::<F>());
    let pcounts = pool.acquire(nb * k * size_of::<u32>());
    let sums = pool.acquire(kd * size_of::<F>());
    let counts = pool.acquire(k * size_of::<u32>());

    let client = pool.client().clone();

    // SAFETY: validated element counts; both kernels bounds-check their unit id
    // against the launched totals and clamp the block row range to `n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let l_arg = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
    let pc_arg = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let (count1, dim1) = launch_dims_1d(psums_len);
    centroid_sumcount_blocked::launch::<F, ActiveRuntime>(
        &client,
        count1,
        dim1,
        x_arg,
        l_arg,
        ps_arg,
        pc_arg,
        n as u32,
        d as u32,
        k as u32,
        nb as u32,
        rpb as u32,
    );

    let ps_arg2 = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
    let pc_arg2 = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let s_arg = unsafe { ArrayArg::from_raw_parts(sums.clone(), kd) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(counts.clone(), k) };
    let (count2, dim2) = launch_dims_1d(kd);
    centroid_reduce_partials::launch::<F, ActiveRuntime>(
        &client,
        count2,
        dim2,
        ps_arg2,
        pc_arg2,
        s_arg,
        c_arg,
        d as u32,
        k as u32,
        nb as u32,
    );

    let sums_dev = DeviceArray::<ActiveRuntime, F>::from_raw(sums, kd);
    let counts_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(counts, k);
    let sums_host: Vec<f64> = sums_dev.to_host(pool).iter().map(|&s| host_to_f64(s)).collect();
    let counts_host: Vec<i64> = counts_dev.to_host(pool).iter().map(|&c| c as i64).collect();

    pool.release(psums, psums_len * size_of::<F>());
    pool.release(pcounts, nb * k * size_of::<u32>());
    sums_dev.release_into(pool);
    counts_dev.release_into(pool);

    Ok((sums_host, counts_host))
}

/// Shared-memory variant of [`centroid_sums_dev`] (see [`use_shared_sums`]):
/// one 64-thread cube per row block holds the whole `k × d` partial in its
/// fixed 4096-slot `SharedMemory` tile with single-writer column ownership —
/// O(n·d) work, no atomics, deterministic. Counts come from the
/// [`count_blocked`] pass on the SAME block layout, and both partial sets fold
/// through the shared [`centroid_reduce_partials`].
fn centroid_sums_shared<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    labels: &DeviceArray<ActiveRuntime, u32>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<(Vec<f64>, Vec<i64>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    let kd = k * d;
    debug_assert!(kd <= 4096, "shared sums caller must gate kd <= 4096");
    let nb_cap = ((8usize << 20) / kd.max(1)).max(64);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);

    let psums_len = nb * kd;
    let psums = pool.acquire(psums_len * size_of::<F>());
    let pcounts = pool.acquire(nb * k * size_of::<u32>());
    let sums = pool.acquire(kd * size_of::<F>());
    let counts = pool.acquire(k * size_of::<u32>());

    let client = pool.client().clone();

    // Stage 1a: shared-memory block sums (one 64-thread cube per block).
    // SAFETY: validated element counts (caller); kernels bounds-check unit ids
    // and clamp block row ranges to `n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let l_arg = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
    let (cc, cd) = launch_cubes_64(nb);
    centroid_sumcount_shared::launch::<F, ActiveRuntime>(
        &client,
        cc,
        cd,
        x_arg,
        l_arg,
        ps_arg,
        n as u32,
        d as u32,
        k as u32,
        nb as u32,
        rpb as u32,
    );

    // Stage 1b: per-block counts on the same layout.
    let l_arg2 = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let pc_arg = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let (c1, d1) = launch_dims_1d(nb * k);
    count_blocked::launch::<ActiveRuntime>(
        &client, c1, d1, l_arg2, pc_arg, n as u32, k as u32, nb as u32, rpb as u32,
    );

    // Stage 2: fold both partial sets (the gather path's reducer).
    let ps_arg2 = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
    let pc_arg2 = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let s_arg = unsafe { ArrayArg::from_raw_parts(sums.clone(), kd) };
    let c_arg = unsafe { ArrayArg::from_raw_parts(counts.clone(), k) };
    let (c2, d2) = launch_dims_1d(kd);
    centroid_reduce_partials::launch::<F, ActiveRuntime>(
        &client, c2, d2, ps_arg2, pc_arg2, s_arg, c_arg, d as u32, k as u32, nb as u32,
    );

    let sums_dev = DeviceArray::<ActiveRuntime, F>::from_raw(sums, kd);
    let counts_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(counts, k);
    let sums_host: Vec<f64> = sums_dev.to_host(pool).iter().map(|&s| host_to_f64(s)).collect();
    let counts_host: Vec<i64> = counts_dev.to_host(pool).iter().map(|&c| c as i64).collect();

    pool.release(psums, psums_len * size_of::<F>());
    pool.release(pcounts, nb * k * size_of::<u32>());
    sums_dev.release_into(pool);
    counts_dev.release_into(pool);

    Ok((sums_host, counts_host))
}

/// GEMM variant of [`centroid_sums_dev`]: `sums(k × d) = onehot(labels)ᵀ · X`
/// on the tiled matmul (the one-hot expansion is an element-wise device
/// kernel; the counts come from the row-blocked [`count_blocked`] +
/// [`count_reduce`] pair). Same readback contract: small host sums + counts.
fn centroid_sums_gemm<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    labels: &DeviceArray<ActiveRuntime, u32>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<(Vec<f64>, Vec<i64>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    guard_u32("n*k", n * k)?;
    let client = pool.client().clone();
    let kd = k * d;

    // One-hot expansion of the labels (n × k, element-wise).
    let oh_len = n * k;
    let onehot =
        DeviceArray::<ActiveRuntime, F>::from_raw(pool.acquire(oh_len * size_of::<F>()), oh_len);
    // SAFETY: validated element counts; kernels bounds-check their unit ids.
    let l_arg = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let oh_arg = unsafe { ArrayArg::from_raw_parts(onehot.handle().clone(), oh_len) };
    let (c1, d1) = launch_dims_1d(oh_len);
    onehot_from_labels::launch::<F, ActiveRuntime>(&client, c1, d1, l_arg, oh_arg, n as u32, k as u32);

    // sums = onehotᵀ (k × n) · X (n × d) — transa reads the stored (n, k)
    // buffer as its transpose with no transpose copy.
    let sums_dev = gemm::<F>(pool, &onehot, (k, n), x, (n, d), true, false, None)?;

    // Counts: row-blocked per-centroid count + tiny fold.
    let nb = n.div_ceil(256).clamp(1, 256);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);
    let pcounts = pool.acquire(nb * k * size_of::<u32>());
    let counts = pool.acquire(k * size_of::<u32>());
    let l_arg2 = unsafe { ArrayArg::from_raw_parts(labels.handle().clone(), n) };
    let pc_arg = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let (c2, d2) = launch_dims_1d(nb * k);
    count_blocked::launch::<ActiveRuntime>(
        &client, c2, d2, l_arg2, pc_arg, n as u32, k as u32, nb as u32, rpb as u32,
    );
    let pc_arg2 = unsafe { ArrayArg::from_raw_parts(pcounts.clone(), nb * k) };
    let cnt_arg = unsafe { ArrayArg::from_raw_parts(counts.clone(), k) };
    let (c3, d3) = launch_dims_1d(k);
    count_reduce::launch::<ActiveRuntime>(&client, c3, d3, pc_arg2, cnt_arg, k as u32, nb as u32);

    let sums_host: Vec<f64> = sums_dev.to_host(pool).iter().map(|&s| host_to_f64(s)).collect();
    let counts_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(counts, k);
    let counts_host: Vec<i64> = counts_dev.to_host(pool).iter().map(|&c| c as i64).collect();
    debug_assert_eq!(sums_host.len(), kd);

    onehot.release_into(pool);
    sums_dev.release_into(pool);
    counts_dev.release_into(pool);
    pool.release(pcounts, nb * k * size_of::<u32>());

    Ok((sums_host, counts_host))
}

/// Count positions where two DEVICE-resident `u32` label buffers differ (the
/// strict `array_equal` convergence check, Pitfall 6). Row-blocked device
/// count + a tiny per-block readback; `0` ⇒ the labeling did not change.
pub fn labels_changed(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, u32>,
    b: &DeviceArray<ActiveRuntime, u32>,
    n: usize,
) -> Result<u64, PrimError> {
    if a.len() != n || b.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: a.len().min(b.len()),
        });
    }
    guard_u32("n", n)?;

    let nb = n.div_ceil(256).clamp(1, 256);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);

    let out = pool.acquire(nb * size_of::<u32>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(nb);
    // SAFETY: validated element counts; the kernel bounds-checks its block id
    // and clamps the row range to `n`.
    let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), n) };
    let b_arg = unsafe { ArrayArg::from_raw_parts(b.handle().clone(), n) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), nb) };
    labels_diff_blocked::launch::<ActiveRuntime>(
        &client,
        count,
        dim,
        a_arg,
        b_arg,
        o_arg,
        n as u32,
        nb as u32,
        rpb as u32,
        0u32,
    );

    let out_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(out, nb);
    let parts = out_dev.to_host(pool);
    out_dev.release_into(pool);
    Ok(parts.iter().map(|&c| c as u64).sum())
}

/// Sum a DEVICE-resident length-`n` `F` vector to a host `f64` scalar via the
/// row-blocked [`block_sum_f`] partial + a tiny per-block readback (folds the
/// per-row assigned-center distances into the inertia without an `n` readback).
pub fn sum_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    v: &DeviceArray<ActiveRuntime, F>,
    n: usize,
) -> Result<f64, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if v.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "v",
            rows: n,
            cols: 1,
            len: v.len(),
        });
    }
    guard_u32("n", n)?;

    let nb = n.div_ceil(256).clamp(1, 256);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);

    let out = pool.acquire(nb * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(nb);
    // SAFETY: validated element counts; the kernel bounds-checks its block id.
    let v_arg = unsafe { ArrayArg::from_raw_parts(v.handle().clone(), n) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), nb) };
    block_sum_f::launch::<F, ActiveRuntime>(
        &client, count, dim, v_arg, o_arg, n as u32, nb as u32, rpb as u32,
    );

    let out_dev = DeviceArray::<ActiveRuntime, F>::from_raw(out, nb);
    let parts = out_dev.to_host(pool);
    out_dev.release_into(pool);
    Ok(parts.iter().map(|&p| host_to_f64(p)).sum())
}

/// Gather `k` rows of the device-resident `x` (`n × d`) by index into a fresh
/// `k × d` device array (the k-means++ init centers) — no host round-trip of
/// `x`.
pub fn gather_rows_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    idx: &[u32],
    n: usize,
    d: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let k = idx.len();
    validate_geometry(x.len(), n, d, k)?;
    for &i in idx {
        if (i as usize) >= n {
            return Err(PrimError::ShapeMismatch {
                operand: "idx",
                rows: i as usize,
                cols: 1,
                len: n,
            });
        }
    }

    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, idx);
    let out = pool.acquire(k * d * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(k * d);
    // SAFETY: validated element counts + index domain; the kernel bounds-checks
    // its unit id against `k·d`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let i_arg = unsafe { ArrayArg::from_raw_parts(idx_dev.handle().clone(), k) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), k * d) };
    gather_rows_idx::launch::<F, ActiveRuntime>(
        &client, count, dim, x_arg, i_arg, o_arg, d as u32, k as u32,
    );
    idx_dev.release_into(pool);
    Ok(DeviceArray::from_raw(out, k * d))
}

/// `mean(var(X, axis=0))` (population variance, ddof=0 — the sklearn tol
/// scaling, Pitfall 6) computed on the DEVICE via a numerically-safe two-pass
/// blocked column reduction ([`col_sum_blocked`] → host means →
/// [`col_sqdiff_blocked`]). Only the small `nblocks × d` partials are read
/// back — never the `n × d` sample matrix.
pub fn feature_mean_var<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<f64, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    guard_u32("n", n)?;
    guard_u32("d", d)?;

    let nb_cap = ((4usize << 20) / d.max(1)).max(64);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);
    let plen = nb * d;

    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(plen);

    // Pass 1: blocked column sums → host means (f64 fold of the partials).
    let psums = pool.acquire(plen * size_of::<F>());
    // SAFETY: validated element counts; the kernels bounds-check their unit id
    // and clamp block row ranges to `n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), plen) };
    col_sum_blocked::launch::<F, ActiveRuntime>(
        &client,
        count.clone(),
        dim,
        x_arg,
        ps_arg,
        n as u32,
        d as u32,
        nb as u32,
        rpb as u32,
    );
    let psums_dev = DeviceArray::<ActiveRuntime, F>::from_raw(psums, plen);
    let parts = psums_dev.to_host(pool);
    let inv_n = 1.0_f64 / n as f64;
    let mut means = vec![0.0_f64; d];
    for b in 0..nb {
        for j in 0..d {
            means[j] += host_to_f64(parts[b * d + j]);
        }
    }
    for m in means.iter_mut() {
        *m *= inv_n;
    }

    // Pass 2: blocked Σ (x − mean)² per column (no E[x²] − E[x]² cancellation).
    let means_f: Vec<F> = means.iter().map(|&m| f64_to_host::<F>(m)).collect();
    let means_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &means_f);
    let x_arg2 = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let m_arg = unsafe { ArrayArg::from_raw_parts(means_dev.handle().clone(), d) };
    let pq_arg = unsafe { ArrayArg::from_raw_parts(psums_dev.handle().clone(), plen) };
    col_sqdiff_blocked::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        x_arg2,
        m_arg,
        pq_arg,
        n as u32,
        d as u32,
        nb as u32,
        rpb as u32,
    );
    let sqparts = psums_dev.to_host(pool);
    let mut var_sum = 0.0_f64;
    for &p in sqparts.iter() {
        var_sum += host_to_f64(p);
    }

    psums_dev.release_into(pool);
    means_dev.release_into(pool);
    Ok((var_sum * inv_n) / d as f64)
}

/// sklearn's `_relocate_empty_clusters_dense` over host-side running sums +
/// counts (T-05-03-02 / CR-01) — shared by [`lloyd_update`] and the
/// device-resident Lloyd loop (which calls it only on the RARE empty-cluster
/// iteration, after reading back `x`/labels/distances on demand).
///
/// Ranks ALL samples by `dist_to_assigned` DESCENDING (stable lowest-index
/// tie-break) and hands the empty clusters the farthest points in order,
/// skipping candidates whose donor would be emptied (`count <= 1`) or which
/// were already relocated; each move fixes the sums (add to new, subtract from
/// donor) and the counts (increment new, decrement donor), so the mean divide
/// never sees a zero count. If some empty cluster has no valid donor left,
/// returns a typed error rather than leaving a center at the origin.
#[allow(clippy::too_many_arguments)]
pub fn relocate_empty_clusters<F>(
    sums_f64: &mut [f64],
    counts_i64: &mut [i64],
    x_host: &[F],
    labels: &[u32],
    dist_to_assigned: &[f64],
    n: usize,
    d: usize,
    k: usize,
) -> Result<(), PrimError>
where
    F: Float + CubeElement + Pod,
{
    let empties: Vec<usize> = (0..k).filter(|&c| counts_i64[c] == 0).collect();
    if empties.is_empty() {
        return Ok(());
    }
    // Farthest-first ranking by (dist DESC, index ASC) — a STRICT total order,
    // so any correct selection yields the same top set. A full O(n log n) sort
    // ran every relocation iteration and dominated relocation-heavy fits
    // (~10ms/iteration at n=500k); instead SELECT a generous top-M prefix in
    // O(n) and sort only that. The prefix walk below can skip candidates
    // (already-relocated / singleton donors), so if it ever exhausts M
    // (pathological), fall back to ranking everything.
    let by_rank = |a: &usize, b: &usize| {
        dist_to_assigned[*b]
            .partial_cmp(&dist_to_assigned[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    let m = (empties.len() * 8 + 64).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    if m < n {
        order.select_nth_unstable_by(m - 1, by_rank);
        order.truncate(m);
    }
    order.sort_by(by_rank);
    let mut relocated: Vec<bool> = vec![false; n];
    let mut cursor = 0usize;
    for &c in empties.iter() {
        let mut moved = false;
        loop {
            while cursor < order.len() {
                let i = order[cursor];
                cursor += 1;
                if relocated[i] {
                    continue;
                }
                let donor = labels[i] as usize;
                // A donor must retain at least one member after losing this
                // point.
                if counts_i64[donor] <= 1 {
                    continue;
                }
                for j in 0..d {
                    let xij = host_to_f64(x_host[i * d + j]);
                    sums_f64[c * d + j] += xij;
                    sums_f64[donor * d + j] -= xij;
                }
                counts_i64[c] += 1;
                counts_i64[donor] -= 1;
                relocated[i] = true;
                moved = true;
                break;
            }
            if moved || order.len() == n {
                break;
            }
            // The walk skipped past the selected top-M prefix (pathological
            // skip density): widen to the FULL ranking. Its first M elements
            // equal the sorted prefix already consumed, so `cursor` stays
            // valid and the walk resumes exactly where it left off.
            order = (0..n).collect();
            order.sort_by(by_rank);
        }
        if !moved {
            // No candidate point can be relocated without emptying its donor
            // (e.g. k == n with fewer than k distinct non-empty donors).
            return Err(PrimError::ShapeMismatch {
                operand: "labels",
                rows: n,
                cols: k,
                len: empties.len(),
            });
        }
    }
    Ok(())
}

/// Validate the shared KMeans geometry (`n * d == x.len()`, `1 <= k <= n`)
/// before any unsafe launch (T-05-03-01 / ASVS V5). A bad `k` surfaces as a
/// `"k"` ShapeMismatch (PrimError has no `InvalidK` variant — the distance.rs /
/// topk.rs convention).
fn validate_geometry(x_len: usize, n: usize, d: usize, k: usize) -> Result<(), PrimError> {
    if n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if k < 1 || k > n {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: n,
        });
    }
    // WR-03: every dimension cast to u32 for the kernel launch geometry (n, d, k)
    // must fit in u32 or the cast silently truncates → an out-of-bounds device
    // read. Reject the overflow as a typed ShapeMismatch BEFORE any launch.
    guard_u32("n", n)?;
    guard_u32("d", d)?;
    guard_u32("k", k)?;
    Ok(())
}

/// WR-03: reject a `usize` dimension that does not fit in the kernel-launch `u32`
/// (an unguarded `dim as u32` truncation becomes an out-of-bounds device read).
fn guard_u32(operand: &'static str, dim: usize) -> Result<(), PrimError> {
    if dim > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand,
            rows: dim,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    Ok(())
}

/// Standard ceiling-division 1D launch config (matches `distance.rs`), folding
/// cube counts past the per-dimension dispatch limit (65535 on wgpu) into the
/// Y dimension — needed for the `n × k` staging-matrix launches. `ABSOLUTE_POS`
/// linearizes over the whole grid and every kernel carries an `if tid < total`
/// guard, so slack cubes are harmless (the `random_forest.rs` shape).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let block = 256u32;
    let cubes = (((n as u32) + block - 1) / block).max(1);
    let y = cubes.div_ceil(MAX_DIM);
    let x = cubes.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// 64-thread workgroup grid for a CUBE-addressed kernel (one cube per row
/// block; folds past the per-dimension dispatch limit like [`launch_dims_1d`];
/// slack cubes are guarded in-kernel — the `random_forest.rs` shape).
fn launch_cubes_64(cubes: usize) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let c = (cubes as u32).max(1);
    let y = c.div_ceil(MAX_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: 64, y: 1, z: 1 },
    )
}

/// 2D launch config for the `dist_combine_clamp` kernel: one unit per output
/// element `(i, j)`, `i` on `ABSOLUTE_POS_X` (rows), `j` on `ABSOLUTE_POS_Y`
/// (cols); 16×16 cubes with in-kernel bounds checks (the `distance.rs` shape).
fn launch_dims_2d(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let bx = 16u32;
    let by = 16u32;
    let cx = ((rows as u32) + bx - 1) / bx;
    let cy = ((cols as u32) + by - 1) / by;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: bx, y: by, z: 1 },
    )
}

// ===========================================================================
// Host-side documented seeded PRNG (ASVS V6 — NEVER OsRng; T-05-03-03)
//
// The `SplitMix64` PRNG was PROMOTED to `crate::prims::rng` (PRIM-06, plan
// 07-02) where it now also backs the Gaussian/Achlioptas/permutation generators.
// `kmeanspp_sample` consumes the SAME struct verbatim via `use
// crate::prims::rng::SplitMix64` (top of this file) — the mix is byte-frozen so
// the k-means++ stream is unchanged (RESEARCH Pitfall 7 / `kmeanspp_test.rs`).
// ===========================================================================

// ===========================================================================
// f32/f64 host bit-cast helpers (promoted to `mlrs_core` — F is f32/f64 only)
// ===========================================================================
