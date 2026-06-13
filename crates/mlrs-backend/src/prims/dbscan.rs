//! `prims::dbscan` — host orchestration for the DBSCAN eps-core-mask primitive
//! (PRIM, CLUSTER-02, D-04).
//!
//! [`eps_core_mask`] is the launch wrapper that builds the `n × n` pairwise
//! SQUARED-distance matrix (the Phase-2 [`distance`] prim, `sqrt = false`),
//! launches the new [`mlrs_kernels::dbscan::eps_core_count`] eps-threshold +
//! per-row count kernel with `eps2 = eps*eps`, then reads the per-row count + the
//! `n × n` adjacency back to host (the `prims::cholesky` tiny-readback idiom,
//! scaled to n²). From the host count it derives `is_core[i] = count[i] >=
//! min_samples`; from the adjacency it builds the host neighbor-list the
//! estimator's index-ordered DFS (plan 07) walks.
//!
//! ## The host readback is the D-04 DOCUMENTED EXCEPTION
//! Distance-style prims keep everything device-resident (the `read_backs == 0`
//! mid-pipeline gate). DBSCAN DELIBERATELY reads the core mask + adjacency back —
//! the cluster expansion is an inherently sequential graph traversal that runs on
//! the host (D-04). This is the SINGLE documented round-trip for the primitive.
//! The `n × n` distance matrix is the dominant allocation; it is BOUNDED and
//! REUSED (the accepted brute-force v1 memory cost — plan 11's DBSCAN memory gate
//! asserts the bound), so the readback does not introduce an unbounded surface.
//!
//! ## Validate BEFORE any unsafe launch (T-05-04-01 / ASVS V5)
//! `n*d == x.len()`, `eps >= 0`, and `min_samples >= 1` are checked and surface a
//! typed [`PrimError`] BEFORE any `ArrayArg::from_raw_parts` / launch — an
//! untrusted geometry / hyperparameter never becomes an out-of-bounds device read.
//! The large-`n` n² allocation is the documented DoS surface (T-05-04-02), bounded
//! by the memory gate.
//!
//! Tests live in `crates/mlrs-backend/tests/dbscan_mask_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::dbscan::eps_core_count;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::distance::distance;
use crate::runtime::ActiveRuntime;

/// The host-side result of the DBSCAN eps-core-mask primitive (D-04 readback):
/// the per-point core mask plus the self-inclusive eps-adjacency the estimator's
/// index-ordered DFS walks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpsCoreMask {
    /// `is_core[i] = (eps-neighbor-count incl. self) >= min_samples`.
    pub is_core: Vec<bool>,
    /// Per-point self-inclusive eps-neighbor count (`count[i] >= 1` always — `i`
    /// is its own neighbor). Length `n`.
    pub counts: Vec<u32>,
    /// Row-major `n × n` self-inclusive eps-adjacency (`adjacency[i*n+j] == true`
    /// iff `d(x_i, x_j) <= eps`). The estimator's DFS expands a cluster by walking
    /// the neighbors of each core point in ascending index order.
    pub adjacency: Vec<bool>,
}

impl EpsCoreMask {
    /// Number of points `n` (`is_core.len()`).
    pub fn n(&self) -> usize {
        self.is_core.len()
    }

    /// The ascending-index neighbor list of point `i` (the columns `j` with
    /// `adjacency[i*n+j]`), self-inclusive — the host DFS frontier (D-04).
    pub fn neighbors(&self, i: usize) -> Vec<usize> {
        let n = self.n();
        let base = i * n;
        (0..n).filter(|&j| self.adjacency[base + j]).collect()
    }
}

/// Compute the DBSCAN eps-core mask + self-inclusive eps-adjacency for the
/// `n × d` row-major point cloud `x` (D-04): the DEVICE builds the `n × n`
/// SQUARED-distance matrix, thresholds it at `eps²`, and counts each point's
/// self-inclusive eps-neighbors; the host reads the count + adjacency back and
/// derives `is_core[i] = count[i] >= min_samples`.
///
/// - `x` is the row-major `n × d` point cloud; `x.len() == n*d` is validated.
/// - `eps >= 0` and `min_samples >= 1` are validated BEFORE any launch
///   (T-05-04-01 / ASVS V5); a violation returns [`PrimError::ShapeMismatch`] (the
///   `distance.rs` convention — `PrimError` carries no dedicated eps/min_samples
///   variant, so a bad hyperparameter surfaces as a `ShapeMismatch` on a synthetic
///   operand name).
/// - Returns an [`EpsCoreMask`] (host core mask + counts + `n × n` adjacency) —
///   the SINGLE documented D-04 round-trip. The host DFS (estimator, plan 07)
///   walks the adjacency; this prim does NOT expand clusters.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn eps_core_mask<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    eps: f64,
    min_samples: u32,
) -> Result<EpsCoreMask, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- T-05-04-01 / ASVS V5: validate geometry + hyperparameters BEFORE any
    //     unsafe launch (large n drives the n² allocation, T-05-04-02). ---
    validate(x.len(), n, d, eps, min_samples)?;

    // --- 1. n² pairwise SQUARED Euclidean distance D = X·X (sqrt=false — the
    //        threshold is `<= eps²` on the squared form, so no boundary sqrt). The
    //        n² matrix is the dominant, bounded, reused allocation (D-04 / plan
    //        11 memory gate). ---
    let d2 = distance::<F>(pool, x, (n, d), x, (n, d), false, None)?;

    // --- 2. Launch the eps-threshold + per-row count kernel: one unit per point,
    //        GATHER (no atomics/SharedMemory). Outputs the n×n adjacency bitmask +
    //        the length-n self-inclusive eps-neighbor count. ---
    let nn = n * n;
    let adj_handle = pool.acquire(nn * size_of::<u32>());
    let count_handle = pool.acquire(n * size_of::<u32>());

    let client = pool.client().clone();
    let (cube_count, cube_dim) = launch_dims_1d(n);

    // eps2 = eps*eps in the kernel's float type (the comparison is on the squared
    // distance, so the squared eps is the threshold). Construct it from the f64
    // value via the bytemuck reinterpret (the covariance.rs:281 idiom) so the f64
    // path keeps full precision — NOT `F::new(f32)`, which would truncate the
    // threshold to f32 and break the f64 oracle's integer-exact count.
    let eps2 = f64_to_f::<F>(eps * eps);

    // SAFETY: lengths are the validated/carried element counts (`d2` is the n×n
    // distance, adj is n×n, count is n); the kernel bounds-checks `i < n` and each
    // unit writes only its own row — mitigates T-05-04-01.
    let d2_arg = unsafe { ArrayArg::from_raw_parts(d2.handle().clone(), nn) };
    let adj_arg = unsafe { ArrayArg::from_raw_parts(adj_handle.clone(), nn) };
    let count_arg = unsafe { ArrayArg::from_raw_parts(count_handle.clone(), n) };

    eps_core_count::launch::<F, ActiveRuntime>(
        &client,
        cube_count,
        cube_dim,
        d2_arg,
        adj_arg,
        count_arg,
        // Scalar args by value in cubecl 0.10 (no ScalarArg — see distance.rs).
        eps2,
        n as u32,
    );

    // --- 3. D-04 DOCUMENTED READBACK: read the per-row count + the n×n adjacency
    //        back to host (the `prims::cholesky` tiny-readback idiom, scaled to
    //        n²). This is the SINGLE deliberate host round-trip — the cluster
    //        expansion is a sequential host graph walk (the estimator's job). ---
    let count_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(count_handle, n);
    let adj_dev = DeviceArray::<ActiveRuntime, u32>::from_raw(adj_handle, nn);
    let counts: Vec<u32> = count_dev.to_host(pool);
    let adj_u32: Vec<u32> = adj_dev.to_host(pool);
    count_dev.release_into(pool);
    adj_dev.release_into(pool);

    // The n² distance scratch is consumed (read by the kernel) and never read
    // again — release it back so `live_bytes` is conserved and the buffer is
    // reusable (the bounded-allocation form, D-04 / plan 11).
    d2.release_into(pool);

    // --- 4. Host-side core decision (kept off the device — the device did only
    //        the n² threshold/count): is_core[i] = count[i] >= min_samples. The
    //        adjacency is widened from the u32 bitmask to bool for the estimator. ---
    let is_core: Vec<bool> = counts.iter().map(|&c| c >= min_samples).collect();
    let adjacency: Vec<bool> = adj_u32.iter().map(|&b| b != 0).collect();

    Ok(EpsCoreMask {
        is_core,
        counts,
        adjacency,
    })
}

/// Validate the eps-core-mask operands (T-05-04-01 / ASVS V5). `x` must be the
/// `n × d` point cloud (`x.len() == n*d`); `eps` must be non-negative and finite;
/// `min_samples` must be `>= 1` (a self-inclusive count is always `>= 1`, so
/// `min_samples == 0` would mark every point core trivially — sklearn requires
/// `min_samples >= 1`). All checks run BEFORE any unsafe launch.
fn validate(
    x_len: usize,
    n: usize,
    d: usize,
    eps: f64,
    min_samples: u32,
) -> Result<(), PrimError> {
    if n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    // WR-03: n (and d, via the distance prim) are cast to u32 for the kernel
    // launch geometry; reject an overflowing dimension BEFORE launch so the cast
    // cannot silently truncate into an out-of-bounds device read.
    if n > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "n",
            rows: n,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    if d > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "d",
            rows: d,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    // eps >= 0 and finite. PrimError has no dedicated InvalidEps variant
    // (distance.rs reports all operand violations as ShapeMismatch), so a bad eps
    // surfaces as a ShapeMismatch on the synthetic "eps" operand.
    if !(eps >= 0.0) || !eps.is_finite() {
        return Err(PrimError::ShapeMismatch {
            operand: "eps",
            rows: 0,
            cols: 0,
            len: 0,
        });
    }
    // min_samples >= 1 (a self-inclusive eps-count is always >= 1).
    if min_samples < 1 {
        return Err(PrimError::ShapeMismatch {
            operand: "min_samples",
            rows: 0,
            cols: 0,
            len: min_samples as usize,
        });
    }
    Ok(())
}

/// Reinterpret a host `f64` as the kernel float type `F` (`f32` / `f64`) without
/// the `F::new(f32)` truncation, so the f64 threshold keeps full precision (the
/// `covariance.rs:281` idiom).
fn f64_to_f<F: Pod>(x: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("dbscan eps_core_mask is f32/f64 only"),
    }
}

/// 1D launch config for `eps_core_count`: ONE unit per point (`ABSOLUTE_POS_X` =
/// `i`), ceiling-division over a 256-wide cube so over-provisioned threads are
/// bounds-checked away in the kernel (matches `distance.rs::launch_dims_1d`).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
