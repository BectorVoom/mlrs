//! Pairwise squared-Euclidean distance host API (PRIM-03) — the GEMM-expansion
//! `‖x_i‖² + ‖y_j‖² − 2·XYᵀ` with an unconditional `max(d², 0)` clamp and an
//! optional sqrt boundary, composing the Plan-01 GEMM and the Plan-02 row
//! squared-norm reduction.
//!
//! ## Why GEMM-expansion (D-07)
//! For `X` (`rows_x × cols`) and `Y` (`rows_y × cols`), the squared Euclidean
//! distance `d²(x_i, y_j) = ‖x_i‖² + ‖y_j‖² − 2·x_i·y_j`. The cross term is the
//! whole `XYᵀ` matrix (one GEMM with `transb=true`); the two norm terms are
//! per-row squared norms `‖x_i‖² = Σ_k X[i,k]²` (the Plan-02 row reduction with
//! [`ScalarOp::SumSq`] — the SQUARED norm, no sqrt). Reusing the validated GEMM
//! and reduction is the single-validated-kernel mandate (one distance serves
//! KMeans, DBSCAN, KNN).
//!
//! ## The clamp produces NO negative distances (Criterion 3 / Pitfall 5)
//! In f32, `‖x_i‖² + ‖y_j‖² − 2·x_i·y_j` for near-identical rows is a
//! catastrophic cancellation that can land slightly negative. The
//! `dist_combine_clamp` kernel applies `max(d², 0)` (STATEMENT form) UNCONDITIONALLY,
//! so the squared distance is never negative and the optional sqrt never sees a
//! negative argument (T-0203-03). The `distance_min_nonnegative` property test
//! pins this on a deliberate cancellation case.
//!
//! ## Squared is the core output; sqrt is the boundary (D-08)
//! [`distance`] returns the clamped SQUARED distance by default; passing
//! `sqrt = true` applies [`sqrt_elem`] in place at the boundary so KNN gets true
//! Euclidean distances. Squaring is the cheaper, sufficient form for the
//! distance-comparison consumers (KMeans/DBSCAN), so sqrt is opt-in.
//!
//! ## Device residency (D-05 / D-10 gate 2)
//! Inputs and every intermediate (`XYᵀ`, the two norm vectors, the clamped
//! output, the optional in-place sqrt) stay on the device as [`DeviceArray`]s.
//! This module performs NO host read-back between stages (the device-residency
//! grep gate over this file is `0`). Scratch + the output buffer are drawn from
//! the [`BufferPool`] (D-11).
//!
//! Tests live in `crates/mlrs-backend/tests/distance_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{
    dist_combine_clamp, euclidean_sq_dist, euclidean_sq_dist_rb, euclidean_sq_dist_rb4,
    euclidean_sq_dist_tiled, sqrt_elem,
};

use crate::capability;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Compute the pairwise squared-Euclidean distance matrix `D` (`rows_x ×
/// rows_y`) between the rows of `x` (`rows_x × cols`) and `y` (`rows_y × cols`)
/// via the GEMM-expansion `‖x_i‖² + ‖y_j‖² − 2·XYᵀ`, clamped to `max(d², 0)`.
///
/// - `x` is the row-major `rows_x × cols` left operand; `y` is `rows_y × cols`.
///   Both share the feature dimension `cols`.
/// - Shapes are validated (`rows_x*cols == x.len()`, `rows_y*cols == y.len()`)
///   BEFORE any launch (D-04 / T-0203-02); a mismatch returns
///   [`PrimError::ShapeMismatch`].
/// - `sqrt = true` applies the optional Euclidean sqrt at the boundary (D-08);
///   `sqrt = false` returns the squared distance (the core output).
/// - The `rows_x × rows_y` result is acquired from `pool` when `out` is `None`,
///   else the supplied buffer is reused (D-11). The result stays device-resident
///   (D-05) — NO host round-trip inside this API.
///
/// ## Internal reduction path (CR-01 / D-03)
/// The per-row SQUARED-norm reductions are an INTERNAL implementation detail of
/// distance, not a caller-visible kernel choice, so they always run on the
/// always-portable [`ReducePath::Shared`] path. The plane (subgroup) path is
/// capability-gated and returns `None` on adapters without subgroup support
/// (e.g. the cpu backend) — forwarding a caller-chosen `Plane` into the norm
/// term would unwrap that `None` and PANIC (the D-03 skip contract is for the
/// reduction's own public callers, not for distance's internal use). Distance
/// therefore exposes NO `path` parameter; the norm reduction is unconditionally
/// shared-path-backed and can never be plane-gated to `None`.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
#[allow(clippy::too_many_arguments)]
pub fn distance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    sqrt: bool,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    distance_with_ynorm::<F>(pool, x, (rows_x, cols), y, (rows_y, cols_y), sqrt, out, None)
}

/// [`distance`], but accepting an ALREADY-COMPUTED `‖y_j‖²` term so a caller that
/// holds `y` fixed across many calls does not recompute it every time (KNN-01).
///
/// ## Why this exists (the measured cost)
/// The `y` squared-norm reduction is `O(rows_y × cols)` and depends on NOTHING
/// about `x`. A blocked/tiled consumer — the brute-force KNN search chunks its
/// query rows so the intermediate distance matrix cannot exceed device memory —
/// calls `distance` once per tile against the SAME training set, so this term is
/// recomputed identically on every tile. With the training set far larger than a
/// tile (the normal case: `n_train` in the hundreds of thousands, a tile a few
/// hundred query rows), that redundant reduction DOMINATES: a tiled run measured
/// a HIGHER cost per tile than an untiled run of the whole problem, which is only
/// possible if a large per-call cost is independent of the tile height.
///
/// Passing `ynorm` hoists it out of the caller's loop. The result is numerically
/// identical — it is the same buffer the internal path would have produced.
///
/// - `ynorm`, when `Some`, must be the `rows_y` per-row SQUARED norms of `y`
///   (i.e. `row_reduce(y, rows_y, cols, ScalarOp::SumSq, ReducePath::Shared)`).
///   Its length is validated before launch; a wrong length is a typed error, not
///   a bad device read.
/// - The caller OWNS a supplied `ynorm` — it is NOT released here, so it survives
///   for the next call. An internally-computed one is released as before.
#[allow(clippy::too_many_arguments)]
pub fn distance_with_ynorm<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    sqrt: bool,
    out: Option<DeviceArray<ActiveRuntime, F>>,
    ynorm: Option<&DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- D-04 / T-0203-02: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(x.len(), (rows_x, cols), y.len(), (rows_y, cols_y), out.as_ref().map(DeviceArray::len))?;
    // A caller-supplied norm term is untrusted geometry like any other operand.
    if let Some(yn) = ynorm {
        if yn.len() != rows_y {
            return Err(PrimError::ShapeMismatch {
                operand: "ynorm",
                rows: rows_y,
                cols: 1,
                len: yn.len(),
            });
        }
    }

    // --- 1. XYᵀ via GEMM(transb=true): (rows_x × cols)·(cols × rows_y) →
    //        rows_x × rows_y. `y` is stored (rows_y × cols); transb reads it as
    //        its transpose (cols × rows_y) with no transpose buffer (D-06). ---
    let xy = gemm::<F>(
        pool,
        x,
        (rows_x, cols),
        y,
        // logical rhs shape (k, n) = (cols, rows_y); transb=true ⇒ stored (rows_y, cols).
        (cols, rows_y),
        false,
        true,
        None,
    )?;

    // --- 2. Per-row SQUARED norms ‖x_i‖² (len rows_x) and ‖y_j‖² (len rows_y)
    //        via the Plan-02 row reduction with SumSq (NO sqrt — distance needs
    //        the squared norm directly). Device-resident outputs. ---
    // CR-01: force the always-portable Shared path for the INTERNAL norm term —
    // never the caller's choice — so the reduction is never plane-gated to None
    // (which would panic the `.expect` below on a non-subgroup adapter, e.g. cpu).
    let xnorm = row_reduce::<F>(pool, x, rows_x, cols, ScalarOp::SumSq, ReducePath::Shared)?
        .expect("shared path is never plane-gated to None");
    // Reuse the caller's precomputed term when supplied (KNN-01); otherwise
    // compute it here exactly as before. `owned_ynorm` is Some ONLY when this
    // call allocated it, which is what decides whether it is released below.
    let owned_ynorm = match ynorm {
        Some(_) => None,
        None => Some(
            row_reduce::<F>(pool, y, rows_y, cols, ScalarOp::SumSq, ReducePath::Shared)?
                .expect("shared path is never plane-gated to None"),
        ),
    };
    let ynorm_ref: &DeviceArray<ActiveRuntime, F> = match (ynorm, &owned_ynorm) {
        (Some(yn), _) => yn,
        (None, Some(yn)) => yn,
        (None, None) => unreachable!("owned_ynorm is Some whenever ynorm is None"),
    };

    // --- 3. Combine + clamp: out[i,j] = max(‖x_i‖² + ‖y_j‖² − 2·XYᵀ[i,j], 0).
    //        Device-resident; the clamp guarantees no negative squared distance
    //        (Criterion 3). Output reuses the caller's buffer (D-11) or a pool
    //        acquisition. ---
    let out_len = rows_x * rows_y;
    let elem = size_of::<F>();
    let out_handle = match &out {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * elem),
    };

    let client = pool.client().clone();
    let (count, dim) = launch_dims_2d(rows_x, rows_y);

    // SAFETY: lengths are the carried DeviceArray element counts (themselves
    // derived from validated host slices); the kernel bounds-checks
    // `i < rows && j < cols` (mitigates T-0203-01).
    let xy_arg = unsafe { ArrayArg::from_raw_parts(xy.handle().clone(), out_len) };
    let xn_arg = unsafe { ArrayArg::from_raw_parts(xnorm.handle().clone(), rows_x) };
    let yn_arg = unsafe { ArrayArg::from_raw_parts(ynorm_ref.handle().clone(), rows_y) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

    dist_combine_clamp::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        xy_arg,
        xn_arg,
        yn_arg,
        out_arg,
        // Scalar args are passed by value in cubecl 0.10 (no ScalarArg wrapper —
        // see spike_test.rs), like `saxpy_kernel`'s `a: F`.
        rows_x as u32,
        rows_y as u32,
    );

    // CR-02 / WR-07: the cross term `xy` (XYᵀ) and the two squared-norm vectors
    // are TRANSIENT scratch — all THREE are consumed by the `dist_combine_clamp`
    // launch above and never read again. The output buffer (`out_handle`) is a
    // SEPARATE handle (the caller's `out`, or a distinct pool acquisition) and is
    // NOT released here — the caller owns the returned result. The optional sqrt
    // pass below reads only `out_handle`, never these three. Release each at its
    // TRUE byte size so `live_bytes` is conserved and the buffers are reusable;
    // the combine kernel's reads are same-stream-ordered before any later reuse.
    xy.release_into(pool);
    xnorm.release_into(pool);
    // Only release the y-norm if THIS call allocated it; a caller-supplied one is
    // owned by the caller and must survive for its next call (KNN-01).
    if let Some(yn) = owned_ynorm {
        yn.release_into(pool);
    }

    // --- 4. Optional Euclidean sqrt at the boundary (D-08), in place over the
    //        already-clamped (non-negative) buffer — so sqrt never sees a
    //        negative argument. Still device-resident. ---
    if sqrt {
        let (scount, sdim) = super::launch_dims_1d(out_len, capability::gather_launch_width());
        let in_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        let sout_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
    }

    // The result stays device-resident (D-05); the caller reads it back via the
    // DeviceArray read-back methods at the boundary when needed.
    Ok(DeviceArray::from_raw(out_handle, out_len))
}

/// Validate distance operand geometry (D-04 / T-0203-02). `x` is `rows_x ×
/// cols`, `y` is `rows_y × cols_y`; the two feature dimensions must agree, and
/// each `rows*cols == len`. The output (if supplied) must be `rows_x × rows_y`.
fn validate_geometry(
    x_len: usize,
    (rows_x, cols): (usize, usize),
    y_len: usize,
    (rows_y, cols_y): (usize, usize),
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    if rows_x.checked_mul(cols).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    if rows_y.checked_mul(cols_y).map(|v| v != y_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: rows_y,
            cols: cols_y,
            len: y_len,
        });
    }
    if cols != cols_y {
        return Err(PrimError::DimMismatch {
            dim: "cols",
            lhs: cols,
            rhs: cols_y,
        });
    }
    if let Some(o) = out_len {
        let expect = rows_x * rows_y;
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out",
                rows: rows_x,
                cols: rows_y,
                len: o,
            });
        }
    }
    Ok(())
}

/// 2D launch config for the `dist_combine_clamp` kernel: one unit per output
/// element `(i, j)`, `i` on `ABSOLUTE_POS_X` (rows), `j` on `ABSOLUTE_POS_Y`
/// (cols). Ceiling-division over a 16×16 cube so over-provisioned threads are
/// bounds-checked away in the kernel.
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

/// Is the untiled (no-data-reuse) distance kernel forced via `MLRS_DIST_UNTILED=1`?
///
/// The shared-memory tiled kernel is the default. This escape hatch exists so the
/// two can be A/B'd on the real target device rather than gated by extrapolation.
fn untiled_distance_forced() -> bool {
    crate::abflag::is_on("MLRS_DIST_UNTILED")
}

/// Is the 1×1 (non-register-blocked) tiled kernel forced via `MLRS_DIST_TILED1X1=1`?
///
/// The 2×2 register-blocked kernel is the default; this selects the previous
/// tiled kernel so the two can be A/B'd on the real target device.
fn tiled_1x1_forced() -> bool {
    crate::abflag::is_on("MLRS_DIST_TILED1X1")
}

/// Is the 2×2 register-blocked kernel forced via `MLRS_DIST_RB2=1`?
///
/// The 4×4 kernel is the default; this selects the 2×2 one for A/B on the real
/// target device.
fn rb2_forced() -> bool {
    crate::abflag::is_on("MLRS_DIST_RB2")
}

/// Which `distance_direct` kernel the ACTIVE BACKEND can actually run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DistVariant {
    /// `euclidean_sq_dist` — no `SharedMemory`, no `sync_cube`, no reuse.
    Untiled,
    /// `euclidean_sq_dist_tiled` — 2 × 256 `F` staged tiles.
    Tiled1x1,
    /// `euclidean_sq_dist_rb` — 2 × 512 `F`, 2×2 register block.
    Rb2,
    /// `euclidean_sq_dist_rb4` — 2 × 2048 `F`, 4×4 register block.
    Rb4,
}

/// Choose the widest [`DistVariant`] whose shared-memory budget fits the active
/// adapter — a CAPABILITY gate, exactly like `knn::fused_topk_applicable`'s.
///
/// ## Why this is required, not an optimization
/// The four kernels need `2 × {2048, 512, 256} × size_of::<F>()` bytes of
/// workgroup storage (16 / 4 / 2 KiB at f32, doubled at f64); the untiled one
/// needs none. cubecl-wgpu's `validate_shared` HARD-ERRORS with
/// `ResourceLimitError::SharedMemory` when a pipeline exceeds
/// `max_compute_workgroup_storage_size`, whose WebGPU downlevel default is
/// exactly 16384. So on a 16 KiB adapter the default 4×4 kernel fails to launch
/// at f32-borderline and always at f64 — and `knn::fused_topk_applicable`
/// rejects the fused path on those same adapters specifically so the two-kernel
/// pipeline can serve them. Without this gate that fallback could not launch
/// either and the capability gate bought nothing.
///
/// ## cpu
/// cpu takes `Untiled` unconditionally. It is the only variant free of
/// `SharedMemory` and `sync_cube`, and on `cubecl-cpu` — one OS thread per unit,
/// spin-barrier `sync_cube` — the barriers are the dominant cost, not the
/// global re-reads the tiling exists to remove. (cubecl-cpu also reports total
/// system RAM as its shared-memory limit, so the byte test below would wave
/// every variant through.)
fn distance_variant<F>() -> DistVariant
where
    F: Float + CubeElement + Pod,
{
    if capability::active_backend_name() == "cpu" {
        return DistVariant::Untiled;
    }
    let limit = capability::active_max_shared_memory();
    let elem = size_of::<F>();
    if limit >= 4096 * elem {
        DistVariant::Rb4
    } else if limit >= 1024 * elem {
        DistVariant::Rb2
    } else if limit >= 512 * elem {
        DistVariant::Tiled1x1
    } else {
        DistVariant::Untiled
    }
}

/// Launch config for the register-blocked kernels: a 16×16 unit cube covers a
/// `block × block` OUTPUT tile, so the cube grid divides each axis by `block`
/// (32 for the 2×2 kernel, 64 for the 4×4 one) rather than by the unit width.
fn launch_dims_blocked(rows: usize, cols: usize, block: u32) -> (CubeCount, CubeDim) {
    let cx = ((rows as u32) + block - 1) / block;
    let cy = ((cols as u32) + block - 1) / block;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: 16, y: 16, z: 1 },
    )
}

/// Pairwise SQUARED-Euclidean distance computed by the DIRECT per-element kernel
/// ([`mlrs_kernels::euclidean_sq_dist`]) rather than the GEMM-expansion (KNN-01).
///
/// Same result as [`distance`] with `sqrt = false`, but it never forms `XYᵀ` and
/// never runs the two norm reductions — see the kernel docs for the measurement
/// that motivated it (the expansion's `cubek-matmul` call is pathologically slow
/// at the tiny contraction dimension a pairwise-distance shape has, and it was
/// ~99% of KNN predict). It is also cancellation-free, so unlike the expansion it
/// needs no `max(d², 0)` clamp to stay non-negative.
///
/// - `x` is the row-major `rows_x × cols` left operand; `y` is `rows_y × cols`.
/// - Geometry is validated BEFORE any `unsafe` launch (D-04 / ASVS V5), reusing
///   the same [`validate_geometry`] as the expansion path.
/// - The `rows_x × rows_y` result is acquired from `pool` when `out` is `None`
///   (D-11) and stays device-resident (D-05).
///
/// Returns the SQUARED distance; apply the boundary sqrt via `top_k`'s `sqrt`
/// flag exactly as with [`distance`].
pub fn distance_direct<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(
        x.len(),
        (rows_x, cols),
        y.len(),
        (rows_y, cols_y),
        out.as_ref().map(DeviceArray::len),
    )?;
    // WR-03: the three dims are cast to u32 for the launch; reject an overflowing
    // dimension BEFORE launch so the cast cannot truncate into a bad bound.
    for (operand, dim) in [("rows_x", rows_x), ("rows_y", rows_y), ("cols", cols)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let out_len = rows_x
        .checked_mul(rows_y)
        .ok_or(PrimError::Overflow {
            operand: "distance_direct",
            lhs: rows_x,
            rhs: rows_y,
        })?;
    let out_handle = match &out {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * size_of::<F>()),
    };

    let client = pool.client().clone();

    // SAFETY: lengths are the carried/validated element counts; both kernels
    // bounds-check `i < rows_x && j < rows_y`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

    // Pick the widest register-blocked kernel the adapter's workgroup storage
    // limit actually admits (see `shared_tiling_budget`). Without this the
    // default 4×4 kernel needs the SAME 16 KiB the fused KNN gate rejects
    // adapters for, so the "fallback" it falls back TO could not launch either.
    let variant = distance_variant::<F>();

    if untiled_distance_forced() || variant == DistVariant::Untiled {
        // A/B escape hatch (`MLRS_DIST_UNTILED=1`): the no-reuse kernel, retained
        // so the tiled one can be measured against it on the real target device.
        // ALSO the automatic choice when no tiled variant fits the adapter's
        // shared-memory limit, and on cpu (it is the only variant with neither
        // `SharedMemory` nor `sync_cube` — see `distance_variant`).
        let (count, dim) = launch_dims_2d(rows_x, rows_y);
        euclidean_sq_dist::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            out_arg,
            rows_x as u32,
            rows_y as u32,
            cols as u32,
        );
    } else if tiled_1x1_forced() || variant == DistVariant::Tiled1x1 {
        // NOTE the axis swap: the tiled kernels map their FASTEST unit index to
        // the output's minor axis (j / rows_y) so the store is coalesced, so
        // CUBE_POS_X must block `rows_y` and CUBE_POS_Y must block `rows_x`.
        let (count, dim) = launch_dims_2d(rows_y, rows_x);
        euclidean_sq_dist_tiled::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            out_arg,
            rows_x as u32,
            rows_y as u32,
            cols as u32,
        );
    } else if rb2_forced() || variant == DistVariant::Rb2 {
        // 2×2 register-blocked: a 16×16 cube covers a 32×32 output block, so the
        // cube grid is over ceil(dim / 32) with the same axis swap as above.
        let (count, dim) = launch_dims_blocked(rows_y, rows_x, 32);
        euclidean_sq_dist_rb::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            out_arg,
            rows_x as u32,
            rows_y as u32,
            cols as u32,
        );
    } else {
        // 4×4 register-blocked: a 16×16 cube covers a 64×64 output block.
        let (count, dim) = launch_dims_blocked(rows_y, rows_x, 64);
        euclidean_sq_dist_rb4::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            out_arg,
            rows_x as u32,
            rows_y as u32,
            cols as u32,
        );
    }

    Ok(DeviceArray::from_raw(out_handle, out_len))
}
