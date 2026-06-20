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

use mlrs_core::PrimError;
use mlrs_kernels::kmeans::{centroid_sumcount, inertia_rows};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::distance::distance;
use crate::prims::rng::SplitMix64;
use crate::runtime::ActiveRuntime;

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

    let empties: Vec<usize> = (0..k).filter(|&c| counts_i64[c] == 0).collect();
    if !empties.is_empty() {
        // sklearn ranks ALL samples by their squared distance to the assigned
        // center and gives the empty clusters the farthest points, in order
        // (`np.argpartition(distances, -n_empty)[-n_empty:]`, then assigned to the
        // empty clusters). We sort the indices by `dist_to_assigned` DESCENDING
        // (stable lowest-index tie-break) and take the first `n_empty` — distinct
        // by construction. For each, move the point from its donor to the empty
        // cluster: fix the sums (add to new, subtract from donor) and the counts
        // (increment new, decrement donor).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            dist_to_assigned[b]
                .partial_cmp(&dist_to_assigned[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        // CR-01: mirror sklearn's `_relocate_empty_clusters_dense` exactly —
        // never drain a donor to empty. Walk the farthest-first ranking and, for
        // each empty cluster, pick the next candidate whose DONOR still has
        // `count >= 2` (so moving its point leaves the donor non-empty) and which
        // has NOT already been relocated. A naive "take order[rank]" can hand the
        // same donor away twice (driving a count to -1) or empty a singleton donor
        // (count 0 → a center silently left at the origin, a WRONG centroid). If
        // no valid donor remains for some empty cluster, surface a typed error
        // rather than leaving a center at the origin.
        let mut relocated: Vec<bool> = vec![false; n];
        let mut cursor = 0usize;
        for &c in empties.iter() {
            let mut moved = false;
            while cursor < n {
                let i = order[cursor];
                cursor += 1;
                if relocated[i] {
                    continue;
                }
                let donor = labels[i] as usize;
                // Skip a candidate whose donor would be emptied (count <= 1) — a
                // donor must retain at least one member after losing this point.
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
            if !moved {
                // No candidate point can be relocated without emptying its donor
                // (e.g. k == n with fewer than k distinct non-empty donors). This
                // is unrecoverable here — never leave a center at the origin.
                return Err(PrimError::ShapeMismatch {
                    operand: "labels",
                    rows: n,
                    cols: k,
                    len: empties.len(),
                });
            }
        }
    }

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

    // Read the samples once to the host so we can build the running per-sample
    // min-D² as centers are added; the per-step D² to the NEW center is computed
    // on-device via the distance prim and read back (INIT-only, D-09c).
    let x_host: Vec<F> = x.to_host(pool);

    // Running nearest-center squared distance per sample (init: distance to the
    // first chosen center, computed on-device for the n-heavy part).
    let mut min_d2: Vec<f64> = device_d2_to_center::<F>(pool, x, n, d, &x_host, chosen[0])?;

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

        // Update the running min-D²: each sample's nearest distance is the min of
        // its current nearest and its distance to the NEW center (device-computed
        // n-heavy term, read back once — D-09c).
        let d2_new = device_d2_to_center::<F>(pool, x, n, d, &x_host, next)?;
        for i in 0..n {
            if d2_new[i] < min_d2[i] {
                min_d2[i] = d2_new[i];
            }
        }
    }

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

/// Compute the squared distance from every sample row to a SINGLE center
/// (`x[center_idx]`) on-device via the [`distance`] prim (`n × 1` result), read
/// back to the host. This is the n-heavy per-sample D² term the k-means++ host
/// draw consumes (INIT-only read-back, D-09c).
fn device_d2_to_center<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    x_host: &[F],
    center_idx: usize,
) -> Result<Vec<f64>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // The center is one row of x; upload it as a 1 × d device array and run the
    // pairwise SQUARED distance (sqrt=false) X(n×d) vs center(1×d) → n × 1.
    let center: Vec<F> = x_host[center_idx * d..(center_idx + 1) * d].to_vec();
    let center_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &center);
    let d2 = distance::<F>(pool, x, (n, d), &center_dev, (1, d), false, None)?;
    let host: Vec<F> = d2.to_host(pool);
    d2.release_into(pool);
    center_dev.release_into(pool);
    Ok(host.iter().map(|&v| host_to_f64(v)).collect())
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

/// Standard ceiling-division 1D launch config (matches `distance.rs`).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
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
// f32/f64 host bit-cast helpers (mirror reduce.rs — F is f32/f64 only)
// ===========================================================================

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side finalize.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans prims are f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans prims are f32/f64 only"),
    }
}
