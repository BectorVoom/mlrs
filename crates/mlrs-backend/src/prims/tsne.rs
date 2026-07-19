//! Host-launch wrapper for the exact-method t-SNE per-iteration device pair
//! (TSNE-01): squared embedding distances ([`squared_distance`], a direct
//! GATHER — NOT the Phase-2 GEMM-expansion `distance` prim, see its docs) →
//! the Student-t affinity GATHER (`tsne_qnum`) → row sums (`tsne_rowsum`, a
//! direct GATHER — NOT `row_reduce`, see its docs) → the KL-gradient GATHER
//! (`tsne_grad`).
//!
//! The device kernels live in the feature-free `mlrs-kernels` crate
//! (`tsne::{tsne_qnum, tsne_grad}`); this layer owns the concrete
//! `ActiveRuntime`, validates geometry HOST-SIDE before any `unsafe` launch
//! (ASVS V5 — the mutual_reachability.rs precedent), and returns the gradient
//! HOST-SIDE (the O(n·d) update rule runs on the host, like the sgd/lbfgs
//! iterative prims) plus the unnormalised-affinity sum `qsum` and the
//! device-resident `qnum` block (the caller reads it back only on the KL
//! error-check iterations, then releases it).
//!
//! Tests live in `crates/mlrs-backend/tests/tsne_test.rs` (AGENTS.md §2): the
//! VALUE oracle asserts the qnum/gradient against a host reference walk.

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use mlrs_kernels::{tsne_grad, tsne_qnum, tsne_rowsum, tsne_sqdist};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Direct squared-Euclidean pairwise distance (`tsne_sqdist` — NOT the
/// GEMM-expansion `distance` prim, see the kernel docs for why). Geometry is
/// validated by every caller before this is reached.
pub fn squared_distance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let nn = n * n;
    let out_handle = pool.acquire(nn * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_2d(n, n);
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), nn) };
    tsne_sqdist::launch::<F, ActiveRuntime>(&client, count, dim, x_arg, o_arg, n as u32, d as u32);
    DeviceArray::from_raw(out_handle, nn)
}

/// sklearn `MACHINE_EPSILON` — `np.finfo(np.double).eps`, the lower clamp on
/// the normalised `Q` (double eps even on the f32 arm, matching sklearn).
pub const MACHINE_EPSILON: f64 = 2.220_446_049_250_313e-16;

/// The 2D cube edge length for the GATHER launches (the knn_graph/
/// mutual_reachability shape). Kernels bounds-check regardless.
const CUBE_DIM_2D: u32 = 16;

/// One exact-method t-SNE gradient evaluation at the current embedding `y`.
pub struct TsneStep<F: Float + CubeElement + Pod> {
    /// The KL gradient (`n × d`, row-major, HOST-side — the O(n·d) sklearn
    /// gains/momentum update consumes it directly).
    pub grad: Vec<F>,
    /// `Σ qnum` over the full matrix (diagonal is 0 by construction), i.e. the
    /// Student-t normaliser `2·Σ_{i<j}`.
    pub qsum: f64,
    /// The dense `n × n` unnormalised affinity block, still device-resident.
    /// Read back only on KL error-check iterations; the CALLER releases it.
    pub qnum: DeviceArray<ActiveRuntime, F>,
}

/// Evaluate the exact t-SNE KL gradient at embedding `y` (`n × d`) against the
/// joint probabilities `p` (dense `n × n`, diagonal 0, device-resident).
///
/// `dof` is `degrees_of_freedom = max(n_components - 1, 1)` (sklearn); the
/// gradient coefficient is `c_f = 2(dof+1)/dof` (`= 4` at the 2-D default).
///
/// Geometry is validated HOST-SIDE BEFORE any launch: `y.len() == n·d`
/// (operand `"y"`), `p.len() == n·n` (operand `"p"`), `n >= 2`, `checked_mul`
/// overflow + `u32`-fit guards on the launch dims.
pub fn tsne_gradient<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y: &DeviceArray<ActiveRuntime, F>,
    p: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    dof: f64,
) -> Result<TsneStep<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: validate geometry HOST-SIDE before any launch. ---
    let nn = n.checked_mul(n).ok_or(PrimError::Overflow {
        operand: "p",
        lhs: n,
        rhs: n,
    })?;
    let nd = n.checked_mul(d).ok_or(PrimError::Overflow {
        operand: "y",
        lhs: n,
        rhs: d,
    })?;
    if n < 2 || y.len() != nd {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: d,
            len: y.len(),
        });
    }
    if p.len() != nn {
        return Err(PrimError::ShapeMismatch {
            operand: "p",
            rows: n,
            cols: n,
            len: p.len(),
        });
    }
    for (operand, dim) in [("y", n), ("y", d)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    // --- 1. Squared embedding distances (direct GATHER, see squared_distance docs). ---
    let dsq = squared_distance::<F>(pool, y, n, d);

    // --- 2. qnum[i,j] = (1 + dsq/dof)^(−(dof+1)/2), diagonal 0. ---
    let elem = size_of::<F>();
    let qnum_handle = pool.acquire(nn * elem);
    let client = pool.client().clone();
    let (count, dim2) = launch_dims_2d(n, n);
    {
        let dsq_arg = unsafe { ArrayArg::from_raw_parts(dsq.handle().clone(), nn) };
        let q_arg = unsafe { ArrayArg::from_raw_parts(qnum_handle.clone(), nn) };
        tsne_qnum::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim2,
            dsq_arg,
            q_arg,
            n as u32,
            f64_to_host::<F>(1.0 / dof),
            f64_to_host::<F>(-(dof + 1.0) / 2.0),
        );
    }
    // dsq is transient scratch — consumed by the qnum launch above.
    dsq.release_into(pool);
    let qnum: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(qnum_handle, nn);

    // --- 3. qsum = Σ qnum (per-row loop kernel + final Σ of n on host). NOT
    //     the generic `row_reduce(Shared)` — its SharedMemory barrier
    //     emulation costs ~11 s per 48×48 call on the cpu runtime (measured);
    //     the plain per-row GATHER is sub-millisecond. ---
    let rowsum_handle = pool.acquire(n * elem);
    {
        let (rcount, rdim) = launch_dims_1d(n);
        let m_arg = unsafe { ArrayArg::from_raw_parts(qnum.handle().clone(), nn) };
        let o_arg = unsafe { ArrayArg::from_raw_parts(rowsum_handle.clone(), n) };
        tsne_rowsum::launch::<F, ActiveRuntime>(&client, rcount, rdim, m_arg, o_arg, n as u32, n as u32);
    }
    let rowsums: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(rowsum_handle, n);
    let qsum: f64 = rowsums
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .sum();
    rowsums.release_into(pool);
    // Guard the degenerate all-collapsed embedding (qsum → 0 would divide by
    // zero; sklearn's Q clamp bounds Q below anyway).
    let qsum_safe = if qsum > MACHINE_EPSILON { qsum } else { MACHINE_EPSILON };

    // --- 4. KL-gradient GATHER; read the n×d gradient back to host. ---
    let grad_handle = pool.acquire(nd * elem);
    {
        let (gcount, gdim) = launch_dims_2d(n, d);
        let p_arg = unsafe { ArrayArg::from_raw_parts(p.handle().clone(), nn) };
        let q_arg = unsafe { ArrayArg::from_raw_parts(qnum.handle().clone(), nn) };
        let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), nd) };
        let g_arg = unsafe { ArrayArg::from_raw_parts(grad_handle.clone(), nd) };
        tsne_grad::launch::<F, ActiveRuntime>(
            &client,
            gcount,
            gdim,
            p_arg,
            q_arg,
            y_arg,
            g_arg,
            n as u32,
            d as u32,
            f64_to_host::<F>(1.0 / qsum_safe),
            f64_to_host::<F>(MACHINE_EPSILON),
            f64_to_host::<F>(2.0 * (dof + 1.0) / dof),
        );
    }
    let grad_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(grad_handle, nd);
    let grad = grad_dev.to_host(pool);
    grad_dev.release_into(pool);

    Ok(TsneStep { grad, qsum: qsum_safe, qnum })
}

/// 1D ceiling-division launch config (one unit per row).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = (((n as u32) + block - 1) / block).max(1);
    (
        CubeCount::Static(cubes, 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// 2D ceiling-division launch config (one unit per output element; the
/// knn_graph/mutual_reachability shape).
fn launch_dims_2d(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let bx = CUBE_DIM_2D;
    let by = CUBE_DIM_2D;
    let cx = ((rows as u32) + bx - 1) / bx;
    let cy = ((cols as u32) + by - 1) / by;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: bx, y: by, z: 1 },
    )
}
