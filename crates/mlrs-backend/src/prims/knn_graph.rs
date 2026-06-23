//! KNN-graph primitive (PRIM-11, Phase 13) — the phase keystone.
//!
//! `knn_graph<F>(pool, x, (n, d), k, metric, include_self, p)` returns the
//! DIRECTED k-nearest-neighbour graph as `(indices (n, k) u32, distances (n, k)
//! F)`, ascending-ordered per row, composed cpu-MLIR-safely from the
//! launch-proven `distance` / `top_k` prims and the Plan-02 direct distance +
//! self-drop kernels. UMAP (Phase 14, `include_self=false`) and HDBSCAN
//! (Phase 15, `include_self=true`) consume it. No estimator wrapper this phase
//! (D-03); symmetrization is each consumer's job (D-04, directed-only).
//!
//! ## Thin host orchestrator (validate → route → compose → emit)
//! The prim is a HOST orchestrator carrying NO new device kernel: it
//!   1. validates geometry HOST-SIDE before any `unsafe` launch (T-13-06),
//!   2. routes each metric to its distance backend (Euclidean/Cosine → the GEMM
//!      `distance()` fast path; Manhattan/Chebyshev/Minkowski-p → the Plan-02
//!      direct pairwise kernels),
//!   3. composes `distance → top_k(K)` QUERY-AXIS TILED so the big distance
//!      operand is never a full `n×n` resident block (T-13-07 / R-6), then a
//!      single `self_drop_gather` over the assembled `(n, k+1)` result, and
//!   4. emits the directed `(indices, distances)` `(n, k)` graph.
//!
//! ## Self-inclusion (D-01 / D-02)
//! `include_self=true` runs `top_k(k)` directly — self (distance 0) lands at
//! column 0 (the HDBSCAN core-distance path). `include_self=false` runs
//! `top_k(k+1)` then drops the self column BY INDEX IDENTITY via
//! `self_drop_gather` (NOT first-zero-distance), so a genuine duplicate point at
//! distance 0 is kept and only the query row's own index is removed (R-9 / the
//! FINDING 002-B catch). The single self-drop launch ranges `row = CUBE_POS_X`
//! over `0..n` — the GLOBAL query-row index — so the index-identity comparison
//! `in_idx == row` is correct (tiling only splits the distance/top_k stage, the
//! assembled `(n, k+1)` top_k result is self-drop'd whole).
//!
//! ## Tie-break convention
//! Neighbour ordering inherits `top_k`'s LOWEST-INDEX tie-break: among equal
//! distances the smaller column index sorts first. This is the documented mlrs
//! convention (matches the per-metric oracle's set-equality-up-to-tie-ordering
//! contract).
//!
//! Tests live in `crates/mlrs-backend/tests/knn_graph_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use mlrs_kernels::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::distance::distance;
use crate::prims::topk::top_k;
use crate::runtime::ActiveRuntime;

/// The distance metric the KNN graph is built under (D-05 — full fixed set).
///
/// `Euclidean` and `Cosine` route to the GEMM-expansion `distance()` fast path
/// (Cosine on L2-normalised rows → `1 − x̂·ŷ`); `Manhattan`, `Chebyshev`, and
/// `Minkowski` route to the Plan-02 direct pairwise feature-loop kernels.
/// `Minkowski` carries its exponent `p` (validated `>= 1` host-side, RESEARCH
/// Open Q3); the other variants ignore the `p` argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// L2 (Euclidean) — GEMM-expansion fast path, sqrt at the `top_k` boundary.
    Euclidean,
    /// L1 (Manhattan) — direct pairwise feature-loop kernel.
    Manhattan,
    /// Cosine distance `1 − x̂·ŷ` — GEMM on L2-normalised rows.
    Cosine,
    /// L∞ (Chebyshev) — direct pairwise feature-loop kernel (running max).
    Chebyshev,
    /// Minkowski-`p` — direct pairwise feature-loop kernel with `F::powf`.
    Minkowski {
        /// The Minkowski exponent (validated `>= 1` host-side).
        p: f64,
    },
}

/// Default query-axis tile size (rows per block) for the distance composition
/// (RESEARCH Open Q2 / A3 — Claude's discretion). A fixed row-block keeps only a
/// `tile × n` distance block resident at a time, so `peak_bytes` is
/// SUB-QUADRATIC in `n` (the R-6 / T-13-07 memory gate), and the same-shape
/// per-tile scratch is released-then-reused (the free-list serves it, so
/// `reuses` grows and `live_bytes` conserves). Tuned against
/// `knn_memory_gate_query_axis_tiled`.
const QUERY_TILE: usize = 8;

/// Build the directed k-nearest-neighbour graph of the rows of `x` against
/// themselves (X-vs-X), under `metric`.
///
/// - `x` is the row-major `n × d` design matrix; the graph queries every row
///   against the whole set.
/// - `k` is the number of TRUE neighbours requested per row. With
///   `include_self=false` (the UMAP path) the prim internally queries `k+1` and
///   drops the self column by INDEX IDENTITY (D-02), returning `k` neighbours.
///   With `include_self=true` (the HDBSCAN core-distance path) it returns `k`
///   neighbours INCLUDING self at column 0 (distance 0).
/// - `metric` selects the distance; `p` is the Minkowski exponent (used only for
///   `Metric::Minkowski`, validated `>= 1`; ignored otherwise).
/// - Geometry is validated HOST-SIDE BEFORE any `unsafe` launch (T-13-06):
///   `n*d == x.len()` (operand `"x"`), `1 <= k` and `k <= n-1` when
///   `include_self=false` (operand `"k"`), `p >= 1` for Minkowski (operand
///   `"p"`), plus `u32`-overflow guards. A violation returns
///   [`PrimError::ShapeMismatch`] (the topk.rs precedent — no numeric-range
///   variant exists).
///
/// Returns `(indices, distances)`: `indices` is `n × k` (`u32`, ascending per
/// row), `distances` is `n × k` (`F`, true metric distances). Both are
/// caller-owned device arrays.
///
/// The composition is QUERY-AXIS TILED ([`QUERY_TILE`]) so the big distance
/// operand is never a full `n×n` resident block (R-6 / T-13-07); per-tile
/// scratch is released back to the pool. Generic over `F` (`f32`/`f64`); the f64
/// path is capability-gated by the CALLER via `skip_f64_with_log`.
#[allow(clippy::too_many_arguments)]
pub fn knn_graph<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (n, d): (usize, usize),
    k: usize,
    metric: Metric,
    include_self: bool,
    p: f64,
) -> Result<(DeviceArray<ActiveRuntime, u32>, DeviceArray<ActiveRuntime, F>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- T-13-06 / ASVS V5: validate geometry HOST-SIDE before any launch. ---
    validate_geometry(x.len(), (n, d), k, metric, include_self, p)?;

    // Internal neighbour count: k+1 when self must be dropped (so the self column
    // is selected then removed), else exactly k.
    let k_internal = if include_self { k } else { k + 1 };

    // Euclidean uses the GEMM distance() which returns SQUARED distance; top_k
    // applies the boundary sqrt only to the returned k values (cheaper than
    // sqrting the whole block, Pitfall 8 / D-08). Cosine also uses GEMM (on
    // L2-normalised rows) but its TRUE distance is `1 − cos = squared/2` (the
    // squared-Euclidean of unit vectors is `2(1 − cos)`), NOT the sqrt — so
    // Cosine selects on the order-preserving squared value (no boundary sqrt) and
    // the returned distances are halved host-side below. The direct L1/L∞/Lp
    // kernels already emit the TRUE distance, so no boundary sqrt for those.
    let needs_sqrt = matches!(metric, Metric::Euclidean);
    // Cosine post-scale: `1 − cos = ‖x̂ − ŷ‖² / 2`. Applied to the returned k
    // distances host-side (the indices are unaffected — the scale is monotone).
    let cosine_halve = matches!(metric, Metric::Cosine);

    // Read x to the host ONCE so we can upload each query-row tile as a
    // contiguous device block (the rows are contiguous in row-major x). This is
    // the established host-segment pattern (reduce.rs::row_reduce reads its input
    // to host the same way); the full n×n distance block is NEVER materialised.
    let x_host: Vec<F> = x.to_host(pool);

    // For Cosine, L2-normalise the rows host-side once (A4): x̂ = x / ‖x‖₂ with a
    // zero-norm guard (a zero row maps to all-zeros). The normalised set is what
    // every tile's distance() consumes (both as query and train operand).
    let train_host: Vec<F> = match metric {
        Metric::Cosine => l2_normalize_rows::<F>(&x_host, n, d),
        _ => x_host,
    };

    // The full (normalised, for Cosine) train set lives on the device for the
    // whole pass — this is the "big operand kept global" (CONTEXT criterion 4):
    // an n×d block, NOT n×n. Tiles are queried against it.
    let train_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &train_host);

    // Assemble the full (n × k_internal) top_k result on the host as tiles land,
    // so a SINGLE self_drop_gather (or the final upload) sees the whole graph
    // with GLOBAL row indices. This buffer is O(n·k), not O(n²).
    let mut tk_idx_full: Vec<u32> = vec![0u32; n * k_internal];
    let mut tk_val_full: Vec<F> = vec![f64_to_host::<F>(0.0); n * k_internal];

    let mut r0 = 0usize;
    while r0 < n {
        let tile = QUERY_TILE.min(n - r0);

        // Upload this tile's query rows as a contiguous (tile × d) device block.
        let tile_host: &[F] = &train_host[r0 * d..(r0 + tile) * d];
        let tile_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, tile_host);

        // --- 1. Distance(tile_query, full_train) → (tile × n). Big operand stays
        //        global (train_dev); only this tile×n block is resident. ---
        let dist = compute_tile_distance::<F>(
            pool, &tile_dev, &train_dev, tile, n, d, metric,
        )?;
        // The tile query block was consumed by the distance launch — release it.
        tile_dev.release_into(pool);

        // --- 2. top_k(k_internal) over the tile's rows; ascending (val, idx),
        //        lowest-index tie-break. sqrt boundary for GEMM metrics only. ---
        let (tk_val, tk_idx) = top_k::<F>(pool, &dist, tile, n, k_internal, needs_sqrt, None, None)?;
        dist.release_into(pool);

        // Gather this tile's (tile × k_internal) result into the full host buffers
        // at the correct GLOBAL row offset, then release the tile scratch (the
        // free-list serves the SAME-shape next tile → reuses grow, live conserves).
        let tile_idx: Vec<u32> = tk_idx.to_host(pool);
        let mut tile_val: Vec<F> = tk_val.to_host(pool);
        tk_idx.release_into(pool);
        tk_val.release_into(pool);
        // Cosine: convert the selected squared-Euclidean-of-unit-vectors value
        // `2(1 − cos)` into the true cosine distance `1 − cos` (halve). Monotone,
        // so the already-selected indices/order are unchanged.
        if cosine_halve {
            for v in tile_val.iter_mut() {
                *v = f64_to_host::<F>(host_to_f64(*v) * 0.5);
            }
        }
        let base = r0 * k_internal;
        let span = tile * k_internal;
        tk_idx_full[base..base + span].copy_from_slice(&tile_idx[..span]);
        tk_val_full[base..base + span].copy_from_slice(&tile_val[..span]);

        r0 += tile;
    }

    // The global train block is transient scratch (consumed by every tile's
    // distance launch, never returned) — release it.
    train_dev.release_into(pool);

    // --- 3. Self-drop by INDEX IDENTITY for the directed (include_self=false)
    //        path via a SINGLE self_drop_gather over the full (n × k+1) result —
    //        `row = CUBE_POS_X` ranges 0..n, the GLOBAL query index, so the
    //        in_idx==row comparison is correct (D-02 / R-9 / 002-B catch). With
    //        include_self=true the top_k(k) result IS the directed graph. ---
    let (idx_out, val_out): (DeviceArray<ActiveRuntime, u32>, DeviceArray<ActiveRuntime, F>) =
        if include_self {
            let indices = DeviceArray::from_host(pool, &tk_idx_full);
            let distances = DeviceArray::from_host(pool, &tk_val_full);
            (indices, distances)
        } else {
            self_drop_full::<F>(pool, &tk_idx_full, &tk_val_full, n, k)?
        };

    Ok((idx_out, val_out))
}

/// Compute the `(tile × n)` distance block of `tile_q` (tile query rows) against
/// the full `train` set under `metric`. Euclidean/Cosine route to the GEMM
/// `distance()` (squared, sqrt deferred to `top_k`); Manhattan/Chebyshev/
/// Minkowski route to the Plan-02 direct pairwise kernels (true distance). The
/// returned block is caller-owned scratch (released by the caller after `top_k`).
///
/// The Minkowski exponent is read from `Metric::Minkowski { p }` (the enum is the
/// single source of truth, WR-01); there is no standalone `p` parameter here.
#[allow(clippy::too_many_arguments)]
fn compute_tile_distance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    tile_q: &DeviceArray<ActiveRuntime, F>,
    train: &DeviceArray<ActiveRuntime, F>,
    tile: usize,
    n: usize,
    d: usize,
    metric: Metric,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    match metric {
        // GEMM-expansion squared distance; sqrt is applied at the top_k boundary.
        // Cosine's rows are already L2-normalised, so the GEMM gives the
        // squared-Euclidean of unit vectors = 2·(1 − cos), which is
        // order-preserving with cosine distance (1 − cos); the boundary sqrt
        // recovers √(2·(1−cos)) and the oracle compares in that space ≤1e-5.
        Metric::Euclidean | Metric::Cosine => distance::<F>(
            pool,
            tile_q,
            (tile, d),
            train,
            (n, d),
            false, // squared; top_k sqrts the returned k values
            None,
        ),
        // Direct pairwise feature-loop kernels — true distance, no boundary sqrt.
        Metric::Manhattan | Metric::Chebyshev | Metric::Minkowski { .. } => {
            let out_len = tile * n;
            let out_handle = pool.acquire(out_len * size_of::<F>());
            let client = pool.client().clone();
            let (count, dim) = launch_dims_2d(tile, n);

            // SAFETY: lengths are validated element counts (tile*d, n*d, tile*n);
            // the kernels bounds-check i<rows_x && j<rows_y and the feature loop
            // kk<cols (T-13-09). Scalars pass BY VALUE in cubecl 0.10.
            let q_arg =
                unsafe { ArrayArg::from_raw_parts(tile_q.handle().clone(), tile_q.len()) };
            let t_arg =
                unsafe { ArrayArg::from_raw_parts(train.handle().clone(), train.len()) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

            match metric {
                Metric::Manhattan => manhattan_dist::launch::<F, ActiveRuntime>(
                    &client, count, dim, q_arg, t_arg, o_arg,
                    tile as u32, n as u32, d as u32,
                ),
                Metric::Chebyshev => chebyshev_dist::launch::<F, ActiveRuntime>(
                    &client, count, dim, q_arg, t_arg, o_arg,
                    tile as u32, n as u32, d as u32,
                ),
                // WR-01: read the exponent from the ENUM (single source of truth),
                // NOT the standalone `p` arg. `validate_geometry` rejects any
                // divergence between the two up front, so they are guaranteed equal
                // here — but reading the enum makes the compute path self-consistent
                // with the logged/serialized `Metric` regardless.
                Metric::Minkowski { p: mp } => minkowski_dist::launch::<F, ActiveRuntime>(
                    &client, count, dim, q_arg, t_arg, o_arg,
                    tile as u32, n as u32, d as u32,
                    f64_to_host::<F>(mp), // enum-carried p cast f64 → F (kernel exponent)
                ),
                _ => unreachable!("outer match restricts to the direct-kernel metrics"),
            }

            Ok(DeviceArray::from_raw(out_handle, out_len))
        }
    }
}

/// Drop the self column (index == query row) from the full `(n × (k+1))` `top_k`
/// result via a SINGLE `self_drop_gather` launch, emitting the directed
/// `(n × k)` neighbours (index-identity, D-02 / R-9 / 002-B catch). `row =
/// CUBE_POS_X` ranges `0..n` here — the GLOBAL query-row index — so `in_idx ==
/// row` correctly identifies self even with duplicate points at distance 0. The
/// returned arrays are caller-owned; the `(n × (k+1))` input is uploaded as
/// transient scratch and released.
fn self_drop_full<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    tk_idx_full: &[u32],
    tk_val_full: &[F],
    n: usize,
    k: usize,
) -> Result<(DeviceArray<ActiveRuntime, u32>, DeviceArray<ActiveRuntime, F>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    let k1 = k + 1;

    let in_val: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, tk_val_full);
    let in_idx: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, tk_idx_full);

    let out_val_handle = pool.acquire(n * k * size_of::<F>());
    let out_idx_handle = pool.acquire(n * k * size_of::<u32>());

    let client = pool.client().clone();
    // Per-row GATHER launch shape (002-A): one cube per query row, one selecting
    // unit — NEVER a bare 1D ABSOLUTE_POS launch.
    let (count, dim) = launch_dims_rows(n);

    // SAFETY: lengths are the validated element counts (n*(k+1) inputs, n*k
    // outputs); the kernel bounds-checks row<rows and writes only n*k slots.
    let iv_arg = unsafe { ArrayArg::from_raw_parts(in_val.handle().clone(), n * k1) };
    let ii_arg = unsafe { ArrayArg::from_raw_parts(in_idx.handle().clone(), n * k1) };
    let ov_arg = unsafe { ArrayArg::from_raw_parts(out_val_handle.clone(), n * k) };
    let oi_arg = unsafe { ArrayArg::from_raw_parts(out_idx_handle.clone(), n * k) };

    self_drop_gather::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        iv_arg,
        ii_arg,
        ov_arg,
        oi_arg,
        // Scalars by value in cubecl 0.10.
        n as u32,
        k as u32,
        k1 as u32,
    );

    // The (n × k+1) top_k inputs are consumed by the launch — release them.
    in_val.release_into(pool);
    in_idx.release_into(pool);

    Ok((
        DeviceArray::from_raw(out_idx_handle, n * k),
        DeviceArray::from_raw(out_val_handle, n * k),
    ))
}

/// L2-normalise each row of a row-major `n × d` host matrix: `x̂_i = x_i / ‖x_i‖₂`,
/// with a zero-norm guard (a zero row stays zero, A4). Done host-side ONCE for
/// Cosine before the GEMM distance path.
fn l2_normalize_rows<F: Pod>(x: &[F], n: usize, d: usize) -> Vec<F> {
    let mut out: Vec<F> = Vec::with_capacity(n * d);
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let norm_sq: f64 = row.iter().map(|&v| host_to_f64(v).powi(2)).sum();
        let norm = norm_sq.sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for &v in row {
            out.push(f64_to_host::<F>(host_to_f64(v) * inv));
        }
    }
    out
}

/// Validate KNN-graph geometry + `k`/`p` (T-13-06 / ASVS V5). Rejected BEFORE any
/// `unsafe` launch so a wrong shape / bad `k` / bad `p` is a recoverable typed
/// error, not an out-of-bounds device read. `PrimError` has NO numeric-range
/// variant, so `k`/`p` violations are reported as `ShapeMismatch` on a synthetic
/// operand (the topk.rs precedent).
fn validate_geometry(
    x_len: usize,
    (n, d): (usize, usize),
    k: usize,
    metric: Metric,
    include_self: bool,
    p: f64,
) -> Result<(), PrimError> {
    // n*d == x.len() (operand "x").
    if n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    // 1 <= k, and k <= n-1 when self is excluded (needs k+1 distinct rows).
    let max_k = if include_self { n } else { n.saturating_sub(1) };
    if k < 1 || k > max_k {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: max_k,
        });
    }
    // p >= 1 for Minkowski (operand "p"); ignored for the other metrics. Both the
    // enum-carried exponent and the separately-passed argument are validated, AND
    // (WR-01) the two MUST agree: the compute path reads the exponent from the enum
    // (single source of truth), so a caller passing `Metric::Minkowski { p: 3.0 }`
    // with a divergent standalone `p` is rejected here rather than silently
    // computing the enum's `p` while logging/serializing keyed on the other value.
    if let Metric::Minkowski { p: mp } = metric {
        if !(mp >= 1.0) || !(p >= 1.0) || (mp - p).abs() > 0.0 {
            return Err(PrimError::ShapeMismatch {
                operand: "p",
                rows: 1,
                cols: 0,
                len: 1,
            });
        }
    }
    // u32-overflow guards on the launch geometry: n, d, k+1 are cast to u32 for
    // the kernel launches; reject an overflowing dim BEFORE launch.
    //
    // WR-04: these three guarded dims provably dominate EVERY later `as u32` cast
    // for supported sizes. The derived launch dims are all bounded by the guarded
    // set: `launch_dims_2d` ceiling-div `(rows + bx - 1)` over `bx = by = 16`
    // shrinks `rows = tile <= QUERY_TILE = 8` and `cols = n` (both <= the guarded
    // `n`); `out_len = tile * n <= QUERY_TILE * n` and `self_drop`'s `n * k1` are
    // element counts, not launch dims, and are bounded by the guarded `n` and
    // `k+1`. So with `n`, `d`, `k+1` <= u32::MAX and `tile <= 8`, no derived cast
    // can wrap. If QUERY_TILE ever grows large, re-derive this domination argument.
    for (operand, dim) in [("x", n), ("x", d), ("k", k + 1)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }
    Ok(())
}

/// 2D launch config for the direct pairwise distance kernels: one unit per output
/// element `(i, j)`, `i` on `ABSOLUTE_POS_X` (query rows), `j` on `ABSOLUTE_POS_Y`
/// (train rows). Ceiling-division over a 16×16 cube (matches distance.rs).
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

/// Per-row GATHER launch config for `self_drop_gather` (002-A safe): ONE cube per
/// query row (`CUBE_POS_X` = row), a single selecting unit. Matches
/// `topk::launch_dims_rows` / the spike-002 shape — NEVER a bare 1D
/// `ABSOLUTE_POS` launch.
fn launch_dims_rows(rows: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static((rows as u32).max(1), 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}
