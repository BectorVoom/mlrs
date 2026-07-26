//! `prims::knn` — host orchestration for the fused KNN neighbor-target gather
//! (KNN-01 perf lever).
//!
//! Launch wrapper for [`mlrs_kernels::knn::knn_regress_mean`]: given the fitted
//! device-resident targets and the `n_query × k` neighbor indices from
//! `prims::topk`, it produces the `n_query` uniform-weight neighbor means WITHOUT
//! leaving the device. It VALIDATES geometry before the `unsafe` launch (ASVS V5,
//! the `prims::topk` precedent) and returns a device-resident result (D-05).
//!
//! It also owns the two brute-force k-nearest search entry points and the
//! dispatch between them:
//!
//! - [`fused_distance_topk`] — the GPU family (shared-memory staging, 256-unit
//!   cubes, `sync_cube`), gated by [`fused_topk_applicable`];
//! - [`cpu_rows_topk`] — the barrier-free cpu row-scan kernels (KNN-04), gated
//!   by [`cpu_rows_applicable`], which callers must check FIRST because on the
//!   cpu backend the fused family is structurally wrong there, not merely
//!   slower (`cubecl-cpu` runs one OS thread per unit and spins on every
//!   barrier).
//!
//! Tests live in `crates/mlrs-backend/tests/knn_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::knn::{
    euclidean_topk_fused, euclidean_topk_fused_dot, euclidean_topk_fused_dot_vec,
    euclidean_topk_fused_dot_vec8, euclidean_topk_rows_cpu, euclidean_topk_rows_cpu_direct,
    euclidean_topk_rows_cpu_vec, knn_regress_mean, row_sumsq, CPU_K_CAP, CPU_MAX_COLS,
    CPU_QUERY_TILE, CPU_QUERY_TILE_WIDE,
};
use mlrs_kernels::topk::select_k_onepass_indexed;
use mlrs_kernels::{copy_elem, copy_elem_cpu_chunked, sqrt_elem};

use crate::capability;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Gather the `k` neighbor targets per query row and return their uniform mean
/// (`n_query` device-resident predictions).
///
/// - `y_train` is the fitted `n_train` regression targets (device-resident).
/// - `idx` is the `n_query × k` row-major `u32` neighbor index buffer from
///   [`crate::prims::topk::top_k`].
/// - Geometry is validated (`idx.len() == n_query * k`, `1 <= k`, `n_train >= 1`,
///   and each dimension fits in `u32`) BEFORE any launch (ASVS V5); a violation
///   returns [`PrimError::ShapeMismatch`].
/// - The `n_query` output is acquired from `pool` and stays device-resident
///   (D-05) — NO host round-trip inside this API.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn knn_regress_mean_gather<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y_train: &DeviceArray<ActiveRuntime, F>,
    idx: &DeviceArray<ActiveRuntime, u32>,
    n_query: usize,
    k: usize,
    n_train: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: validate every geometry the launch derives its bounds from
    //     BEFORE the unsafe launch. ---
    if k < 1 || n_query < 1 {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: n_query,
            cols: k,
            len: idx.len(),
        });
    }
    if n_query.checked_mul(k).map(|v| v != idx.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "idx",
            rows: n_query,
            cols: k,
            len: idx.len(),
        });
    }
    if y_train.len() != n_train || n_train < 1 {
        return Err(PrimError::ShapeMismatch {
            operand: "y_train",
            rows: n_train,
            cols: 1,
            len: y_train.len(),
        });
    }
    // The three scalars are cast to u32 for the launch; reject an overflowing
    // dimension so the cast cannot silently truncate into a bad bound (WR-03).
    for (operand, dim) in [("n_query", n_query), ("k", k), ("n_train", n_train)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let out_handle = pool.acquire(n_query * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n_query);

    // SAFETY: lengths are the validated element counts carried by the arrays (the
    // kernel additionally bounds-checks `q < n_query` and `t < n_train`), NEVER
    // raw caller geometry.
    let y_arg = unsafe { ArrayArg::from_raw_parts(y_train.handle().clone(), y_train.len()) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx.handle().clone(), idx.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n_query) };

    knn_regress_mean::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        y_arg,
        idx_arg,
        out_arg,
        n_query as u32,
        k as u32,
        n_train as u32,
    );

    Ok(DeviceArray::from_raw(out_handle, n_query))
}

/// DEVICE-TO-DEVICE duplicate of `src` into a fresh pool-owned buffer.
///
/// The ownership primitive for fitted state: an estimator that must keep a
/// caller's operand past the `fit` call takes a copy with this instead of
/// cloning the caller's `Handle`. Cloning the handle makes the estimator ALIAS
/// a buffer the caller still owns, and `DeviceArray::release_into` is the
/// repo-wide way to hand a buffer back to the [`BufferPool`] free list — after
/// which the very next `acquire` of the same byte size returns that live
/// training matrix and the estimator silently predicts from overwritten memory.
/// Nothing in the type system stops a caller from doing that with its own
/// array, so the invariant cannot be prose.
///
/// This is NOT the `to_host` → `from_host` round-trip KNN-01 deleted: no bus
/// traffic and no host allocation, just one device pass over the buffer. At the
/// 200k × 32 f32 ladder rung `fit` still measures 0.0000 s.
///
/// KNN-FIT-01: on the **cpu** backend this dispatches
/// [`copy_elem_cpu_chunked`] (one cube of `cpu_launch_units` threads, each
/// copying a contiguous span) instead of [`copy_elem`]'s
/// one-unit-per-element GATHER shape. `copy_elem`'s `ceil(len /
/// cpu_launch_units)`-cube grid pays a per-cube thread-spawn/join cost on
/// `cubecl-cpu` (one OS thread per unit) for every `cpu_launch_units`
/// elements — at `KNeighborsClassifier::fit`'s 100k×32 f32 training-matrix
/// size that measured ~1.9 GB/s, an order of magnitude below a plain memcpy,
/// and made `fit` lose to sklearn on cpu. Other backends are GPU-shaped
/// hardware (many cheap concurrent lanes), so they keep the original
/// one-thread-per-element form.
///
/// Every `#[cube(launch)]` kernel's `Array` metadata is `u32`-indexed (a
/// structural limit of the whole prims layer, e.g. `launch_dims_1d`'s
/// `n as u32` below), so `len` is asserted to fit before either kernel is
/// dispatched — a loud panic instead of the silent truncated-copy this guard
/// replaces (code-review KNN-FIT-01).
pub fn device_copy<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    src: &DeviceArray<ActiveRuntime, F>,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let len = src.len();
    assert!(
        len <= u32::MAX as usize,
        "device_copy: {len} elements exceeds the u32 index ceiling every CubeCL kernel launch is bound by"
    );
    let handle = pool.acquire(len * size_of::<F>());
    let client = pool.client().clone();
    // SAFETY: both lengths are the same validated element count carried by the
    // source array; both kernels bounds-check the thread/tid against it.
    let in_arg = unsafe { ArrayArg::from_raw_parts(src.handle().clone(), len) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(handle.clone(), len) };
    if capability::active_backend_name() == "cpu" {
        let threads = capability::cpu_launch_units();
        let count = CubeCount::Static(1, 1, 1);
        let dim = CubeDim { x: threads, y: 1, z: 1 };
        copy_elem_cpu_chunked::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
    } else {
        let (count, dim) = launch_dims_1d(len);
        copy_elem::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg);
    }
    DeviceArray::from_raw(handle, len)
}

/// Standard ceiling-division 1D launch config (matches `topk.rs::launch_dims_1d`).
///
/// The 256-unit block is a GPU warp-multiple. On the cpu backend a unit is an
/// OS THREAD (`cubecl_cpu::compute::runner`), so a 256-unit launch spawns 256
/// threads per kernel and pays 256 task hand-offs plus a 256-way join — for the
/// three GATHER feeders this helper drives (`row_sumsq`, `knn_regress_mean`,
/// `sqrt_elem`) that overhead can exceed the work itself. On cpu the block is
/// therefore the core count and the CUBE loop supplies the rest of the range;
/// every kernel launched through here indexes by `ABSOLUTE_POS_X` alone, so the
/// split between cube and unit is a pure scheduling choice with no effect on
/// the result.
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = capability::gather_launch_width();
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// Should the CPU-shaped row-scan kernel serve this selection?
///
/// True only on the cpu backend, for the shape
/// [`euclidean_topk_rows_cpu`](mlrs_kernels::knn::euclidean_topk_rows_cpu)
/// covers: `cols <= CPU_MAX_COLS` (its local transposed query cache) and
/// `1 <= k <= min(n_train, CPU_K_CAP)` (its local list stride). Unlike the GPU
/// fused gate there is NO minimum `n_query` — the kernel's parallelism is
/// query TILES over core-count units, so it covers the whole range.
///
/// `MLRS_KNN_CPU_ROWS=0` forces the shared-memory fused family back on for
/// on-target A/B (`=1` cannot force it onto a non-cpu backend: the kernel's
/// one-thread-per-unit shape is meaningless on a GPU).
pub fn cpu_rows_applicable<F>(n_query: usize, n_train: usize, n_features: usize, k: usize) -> bool
where
    F: Float + CubeElement + Pod,
{
    capability::active_backend_name() == "cpu"
        && n_query >= 1
        && n_features >= 1
        && n_features <= CPU_MAX_COLS as usize
        && k >= 1
        && k <= n_train
        && k <= CPU_K_CAP as usize
        && std::env::var("MLRS_KNN_CPU_ROWS").map(|v| v != "0").unwrap_or(true)
}

/// Use the SIMD `euclidean_topk_rows_cpu_vec` kernel (default) or the scalar
/// `euclidean_topk_rows_cpu` one? The two are bitwise-identical, so this is a
/// pure A/B switch (`MLRS_KNN_CPU_VEC=0` picks scalar).
fn cpu_rows_vec_selected() -> bool {
    std::env::var("MLRS_KNN_CPU_VEC").map(|v| v != "0").unwrap_or(true)
}

/// Should the cpu path use the CANCELLATION-FREE kernel? DEFAULT YES — the one
/// place the cpu arm's default differs from the GPU family's, and it is a
/// MEASURED difference on each target, not an extrapolation.
///
/// The expansion `‖x‖² + ‖y‖² − 2·x·y` is a difference of large numbers, so in
/// f32 it can order two near-equidistant candidates differently from the exact
/// answer. Measured at the `n_train = 100_000, d = 32, k = 10` ladder rung: 1
/// query row in 10_000 took a 10th neighbour 0.24 f32 ULP further away than
/// sklearn's, moving that row's prediction by 0.105 — twenty times the
/// project's 1e-5 agreement bar. With this kernel the worst disagreement over
/// the whole ladder is 9.5e-7.
///
/// On a GPU the expansion is worth that risk: it halves the inner-loop
/// instruction count, which is the bound there. On cpu it is NOT — the bound is
/// IR-op count at LLVM `-O0`, where one extra vector subtract per feature costs
/// only 3-12% end to end (0.358 s vs 0.395 s at the 100k rung, 1.48 s vs 1.66 s
/// at 200k), and both arms still beat sklearn's brute path on every rung. Buying
/// correctness at 10% is the project's stated trade ("if everything else fails,
/// the numerical results must be right").
///
/// `MLRS_KNN_DOT=1` forces the expansion back for A/B; `MLRS_KNN_DIRECT=1` is
/// accepted for symmetry with the GPU path's switch and is a no-op here.
fn direct_selected() -> bool {
    if std::env::var("MLRS_KNN_DIRECT").map(|v| v == "1").unwrap_or(false) {
        return true;
    }
    if std::env::var("MLRS_KNN_DOT").map(|v| v == "1").unwrap_or(false) {
        return false;
    }
    true
}

/// Launch config for the cpu row-scan kernels: one unit per `tile`-row query
/// block, `cpu_units()` units per cube, and as many cubes as it takes to cover
/// the query set.
///
/// Keeping the UNIT count at the core count (rather than the tile count) is the
/// whole point: the units are the threads, the cubes are their serial loop, and
/// all units advance through the same cube index together — so the training
/// rows they stream at any moment are the same ones, and `cpu_units() * tile`
/// query rows share every training cache line pulled in.
///
/// `tile` is the CALLING KERNEL's query-tile width, which is part of that
/// kernel's type (its `Vector` lane count) and is NOT the same for all of them:
/// the DIRECT default runs [`CPU_QUERY_TILE_WIDE`], the expansion arms
/// [`CPU_QUERY_TILE`]. Passing it in is what keeps the grid and the kernel from
/// drifting — a mismatch would silently leave query rows unwritten.
fn cpu_rows_launch_dims(n_query: usize, tile: u32) -> (CubeCount, CubeDim) {
    let units = capability::cpu_launch_units();
    let tiles = (n_query as u32).div_ceil(tile);
    let cubes = tiles.div_ceil(units).max(1);
    (
        CubeCount::Static(cubes, 1, 1),
        CubeDim { x: units, y: 1, z: 1 },
    )
}

/// CPU-shaped fused k-nearest search (KNN-04): norms + one barrier-free
/// row-scan kernel + the boundary sqrt.
///
/// Same results as [`fused_distance_topk`]'s dot path (bitwise — see the kernel
/// docs), with none of its shared-memory staging or `sync_cube` rendezvous,
/// which `cubecl-cpu` implements as spin barriers across one-thread-per-unit.
///
/// Dispatch through [`cpu_rows_applicable`] first — this function validates the
/// caps it indexes with but assumes the caller already matched geometry
/// (`n_query * n_features == xq.len()`, likewise for the training matrix).
/// `crates/mlrs-backend/tests/knn_test.rs` gates it against
/// [`fused_distance_topk`].
#[allow(clippy::too_many_arguments)]
pub fn cpu_rows_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, n_features): (usize, usize),
    x_train: &DeviceArray<ActiveRuntime, F>,
    n_train: usize,
    k: usize,
    sqrt: bool,
) -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, u32>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: validate every bound the launch derives from. ---
    if n_query.checked_mul(n_features).map(|v| v != xq.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_query,
            cols: n_features,
            len: xq.len(),
        });
    }
    if n_train.checked_mul(n_features).map(|v| v != x_train.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: n_features,
            len: x_train.len(),
        });
    }
    if k < 1 || k > n_train || k > CPU_K_CAP as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: n_train.min(CPU_K_CAP as usize),
        });
    }
    if n_features < 1 || n_features > CPU_MAX_COLS as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "features",
            rows: n_features,
            cols: 0,
            len: CPU_MAX_COLS as usize,
        });
    }
    for (operand, dim) in [("rows", n_query), ("cols", n_train)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let out_len = n_query * k;
    let client = pool.client().clone();

    // CANCELLATION-FREE arm (`MLRS_KNN_DIRECT=1`, the same escape hatch the GPU
    // dot path honours): `Σ (x − y)²` instead of the expansion, one extra vector
    // op per feature but immune to the near-tie mis-ordering the expansion has
    // in f32. It needs no norms, so the two feeder launches are skipped too.
    if direct_selected() {
        let val_handle = pool.acquire(out_len * size_of::<F>());
        let idx_handle = pool.acquire(out_len * size_of::<u32>());
        let (count, dim) = cpu_rows_launch_dims(n_query, CPU_QUERY_TILE_WIDE);
        let x_arg = unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), xq.len()) };
        let y_arg = unsafe { ArrayArg::from_raw_parts(x_train.handle().clone(), x_train.len()) };
        let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), out_len) };
        euclidean_topk_rows_cpu_direct::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            val_arg,
            idx_arg,
            n_query as u32,
            n_train as u32,
            n_features as u32,
            k as u32,
        );
        if sqrt {
            let (scount, sdim) = launch_dims_1d(out_len);
            let in_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
            let sout_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
            sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
        }
        return Ok((
            DeviceArray::from_raw(val_handle, out_len),
            DeviceArray::from_raw(idx_handle, out_len),
        ));
    }

    // Per-row squared norms for the `‖x‖² + ‖y‖² − 2·x·y` expansion — the same
    // one-launch GATHER feeder the GPU dot path uses.
    let xnorm_handle = pool.acquire(n_query * size_of::<F>());
    let ynorm_handle = pool.acquire(n_train * size_of::<F>());
    {
        let (nc, nd) = launch_dims_1d(n_query);
        let in_arg = unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), xq.len()) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(xnorm_handle.clone(), n_query) };
        row_sumsq::launch::<F, ActiveRuntime>(
            &client, nc, nd, in_arg, out_arg, n_query as u32, n_features as u32,
        );
        let (nc, nd) = launch_dims_1d(n_train);
        let in_arg = unsafe { ArrayArg::from_raw_parts(x_train.handle().clone(), x_train.len()) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(ynorm_handle.clone(), n_train) };
        row_sumsq::launch::<F, ActiveRuntime>(
            &client, nc, nd, in_arg, out_arg, n_train as u32, n_features as u32,
        );
    }

    let val_handle = pool.acquire(out_len * size_of::<F>());
    let idx_handle = pool.acquire(out_len * size_of::<u32>());
    let (count, dim) = cpu_rows_launch_dims(n_query, CPU_QUERY_TILE);

    // SAFETY: lengths are the carried/validated element counts; the kernel
    // bounds-checks every query row against `rows_x` and never indexes `y`
    // past `rows_y * cols`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), xq.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(x_train.handle().clone(), x_train.len()) };
    let xn_arg = unsafe { ArrayArg::from_raw_parts(xnorm_handle.clone(), n_query) };
    let yn_arg = unsafe { ArrayArg::from_raw_parts(ynorm_handle.clone(), n_train) };
    let val_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), out_len) };

    // SIMD lane-per-query-row form by default; `MLRS_KNN_CPU_VEC=0` forces the
    // scalar kernel (bitwise-identical results, so this is a pure A/B switch).
    if cpu_rows_vec_selected() {
        euclidean_topk_rows_cpu_vec::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            xn_arg,
            yn_arg,
            val_arg,
            idx_arg,
            n_query as u32,
            n_train as u32,
            n_features as u32,
            k as u32,
        );
    } else {
        euclidean_topk_rows_cpu::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            xn_arg,
            yn_arg,
            val_arg,
            idx_arg,
            n_query as u32,
            n_train as u32,
            n_features as u32,
            k as u32,
        );
    }

    DeviceArray::<ActiveRuntime, F>::from_raw(xnorm_handle, n_query).release_into(pool);
    DeviceArray::<ActiveRuntime, F>::from_raw(ynorm_handle, n_train).release_into(pool);

    // Euclidean sqrt over ONLY the returned k values per row (D-08).
    if sqrt {
        let (scount, sdim) = launch_dims_1d(out_len);
        let in_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        let sout_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
    }

    Ok((
        DeviceArray::from_raw(val_handle, out_len),
        DeviceArray::from_raw(idx_handle, out_len),
    ))
}

/// Local-list capacity of [`euclidean_topk_fused`] (must match the kernel's
/// comptime 16-slot list stride). Selections with `k` beyond this use the
/// two-kernel `distance → top_k` path.
pub const FUSED_K_CAP: usize = 16;

/// Query rows covered per cube by the fused kernel (`CUBE_POS_Y * 64`).
const FUSED_QUERY_BLOCK: usize = 64;

/// The fused kernel's grid is `(1, ceil(n_query / 64), 1)`; CUDA caps grid.y at
/// 65535, so this is the largest fusable `n_query`.
const FUSED_MAX_QUERY: usize = 65_535 * FUSED_QUERY_BLOCK;

/// Is the fused distance+top-k kernel forced OFF via `MLRS_KNN_UNFUSED=1`?
///
/// The fused kernel is the default wherever [`fused_topk_applicable`] holds.
/// This escape hatch exists ONLY so the fused and two-kernel paths can be A/B'd
/// on the real target device (a perf kernel must never be gated onto a backend
/// by extrapolating from a different backend's numbers).
fn fused_topk_forced_off() -> bool {
    std::env::var("MLRS_KNN_UNFUSED").map(|v| v == "1").unwrap_or(false)
}

/// Should the DOT-EXPANSION fused kernel be used (when `n_features <= 32`)?
///
/// Backend-aware default, each side MEASURED on its real target (never
/// extrapolated). Since KNN-03 the dot family includes the vectorized
/// variants, and the WARM-ladder A/B has dot+vec8 ahead of the direct kernel
/// on BOTH targets (local wgpu RADV: 0.103/0.321 s vs direct 0.146/0.401 at
/// 50k/100k; Kaggle T4 r3: 0.062 vs 0.108 at 100k) — so every GPU backend now
/// defaults to the dot family. cpu keeps it too (scalar variant below).
/// `MLRS_KNN_DIRECT=1` forces direct anywhere, `MLRS_KNN_DOT=1` forces dot
/// anywhere — the two A/B escape hatches.
fn dot_fused_selected() -> bool {
    if std::env::var("MLRS_KNN_DOT").map(|v| v == "1").unwrap_or(false) {
        return true;
    }
    if std::env::var("MLRS_KNN_DIRECT").map(|v| v == "1").unwrap_or(false) {
        return false;
    }
    true
}

/// Which dot-family kernel variant to launch (KNN-03).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DotVariant {
    /// Scalar shared reads — the KNN-02 kernel.
    Scalar,
    /// `Vector<F,4>` tiles, 4×4 output tile per unit (2 loads / 16 FMAs).
    Vec4,
    /// `Vector<F,4>` tiles, 4×8 output tile per unit (3 loads / 32 FMAs).
    Vec8,
}

/// Select the dot-kernel variant for `n_features` (KNN-03 r3 defaults, every
/// arm MEASURED on its real target — never extrapolated):
///
/// - **cuda/rocm** (Kaggle T4, warm best-of-3): vec8 wins at d=32 (0.062 vs
///   vec4 0.0745 at 100k) but loses ~10% at d=16 (0.0192 vs 0.0175 at 50k) —
///   the per-unit fold is a fixed cost per block and its share grows as `d`
///   shrinks, and the 128-unit cube has half the fold parallelism. So: vec8
///   for `d > 16`, vec4 otherwise.
/// - **wgpu** (AMD RADV, warm): vec8 won at BOTH d=16 and d=32 (0.103/0.321
///   vs vec4 0.135/0.422) — vec8 everywhere.
/// - **cpu**: scalar — the MLIR pipeline runs the scalar kernel correctly and
///   cpu is not a perf target; the vector variants stay test-forced there.
///
/// `MLRS_KNN_VEC8=1/0` and `MLRS_KNN_VEC=1/0` force/deny each vector variant
/// anywhere — the A/B escape hatches (VEC8 is consulted first).
fn dot_variant_selected(n_features: usize) -> DotVariant {
    match std::env::var("MLRS_KNN_VEC8").ok().as_deref() {
        Some("1") => return DotVariant::Vec8,
        Some("0") => {}
        _ => match capability::active_backend_name() {
            "cuda" | "rocm" => {
                if n_features > 16 {
                    return DotVariant::Vec8;
                }
            }
            "wgpu" => return DotVariant::Vec8,
            _ => {}
        },
    }
    match std::env::var("MLRS_KNN_VEC").ok().as_deref() {
        Some("1") => DotVariant::Vec4,
        Some("0") => DotVariant::Scalar,
        _ => match capability::active_backend_name() {
            "cuda" | "rocm" | "wgpu" => DotVariant::Vec4,
            _ => DotVariant::Scalar,
        },
    }
}

/// Can [`fused_distance_topk`] run this selection?
///
/// - `k <= FUSED_K_CAP` (the kernel's comptime local-list stride) and
///   `1 <= k <= n_train` (a valid selection at all);
/// - the adapter's workgroup shared-memory limit covers the kernel's budget —
///   `xs + ys` (2 × 2048 `F`) + merge scratch (256 `F` + 512 `u32`): 19 KB at
///   f32, which EXCEEDS a default wgpu adapter's 16 KiB limit, so wgpu takes
///   the two-kernel path on capability grounds (not by extrapolation);
/// - the query count fits the kernel's `(1, grid.y, 1)` launch;
/// - the backend is NOT cpu (see below);
/// - `MLRS_KNN_UNFUSED=1` does not force the A/B escape hatch.
///
/// ## Never on cpu, at any shape
/// `cubecl-cpu`'s `max_shared_memory_size` is reported as TOTAL SYSTEM MEMORY
/// (`cubecl_cpu::runtime`), so the shared-memory gate below is vacuously true
/// there and would wave the 256-unit `sync_cube` kernel straight through. That
/// kernel is structurally wrong on cpu, not merely slower — one OS thread per
/// unit, spin-barrier `sync_cube` — so this returns `false` on cpu
/// UNCONDITIONALLY, including for the shapes
/// [`cpu_rows_applicable`] does not cover (`n_features > CPU_MAX_COLS`,
/// `k > CPU_K_CAP`). Those fall through to the two-kernel `distance_direct →
/// top_k` pipeline, whose `top_k` has its own cpu arm
/// (`prims::topk::cpu_serial_select`).
pub fn fused_topk_applicable<F>(n_query: usize, n_train: usize, k: usize) -> bool
where
    F: Float + CubeElement + Pod,
{
    if capability::active_backend_name() == "cpu" {
        return false;
    }
    // The kernel's ONLY shared allocations are the two staging tiles (the
    // k-round merge reuses them, F-encoding indices/validity): 16 KiB at f32 —
    // exactly a default wgpu adapter's workgroup limit, and 4 cubes per 64 KiB
    // CUDA SM.
    let shared_needed = 4096 * size_of::<F>();
    // The F-encoded merge index must round-trip exactly: idx < 2^24 at f32
    // (every u32 is exact at f64). Brute-force KNN is impractical far below
    // this bound anyway.
    let idx_exact = size_of::<F>() != 4 || n_train <= (1usize << 24);
    k >= 1
        && k <= FUSED_K_CAP
        && k <= n_train
        && n_query >= FUSED_MIN_QUERY
        && n_query <= FUSED_MAX_QUERY
        && idx_exact
        && capability::active_max_shared_memory() >= shared_needed
        && !fused_topk_forced_off()
}

/// Smallest `n_query` the fused kernel is dispatched for. Below this the
/// launch is only `n_query/64` cubes — measured on BOTH targets (Kaggle T4
/// best-of-3: 2000 queries ran 2.3× FASTER on the two-kernel path; the local
/// wgpu adapter agrees directionally), the two-kernel pipeline wins when the
/// fused grid cannot cover the device.
const FUSED_MIN_QUERY: usize = 4096;

/// FUSED brute-force k-nearest search: squared-Euclidean distance + top-k
/// selection in ONE kernel, never materializing the `n_query × n_train`
/// distance matrix (KNN-02).
///
/// Same result as `distance_direct` → `top_k` (bitwise — see the kernel docs),
/// but global traffic drops to the feature reads plus the `n_query × k`
/// output; the intermediate matrix (the dominant traffic term after KNN-01)
/// never exists, and no query tiling is needed because nothing
/// `O(n_query × n_train)` is ever allocated.
///
/// - `xq` is the row-major `n_query × n_features` query matrix; `x_train` is
///   `n_train × n_features`.
/// - Geometry is validated BEFORE any `unsafe` launch (ASVS V5), including
///   `k <= FUSED_K_CAP` — dispatch via [`fused_topk_applicable`] first.
/// - `sqrt = true` applies the Euclidean sqrt to ONLY the returned `rows × k`
///   values (D-08), exactly as `top_k` does.
/// - Both `n_query × k` results are pool-acquired and stay device-resident
///   (D-05).
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
#[allow(clippy::too_many_arguments)]
pub fn fused_distance_topk<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    xq: &DeviceArray<ActiveRuntime, F>,
    (n_query, n_features): (usize, usize),
    x_train: &DeviceArray<ActiveRuntime, F>,
    (n_train, n_features_t): (usize, usize),
    k: usize,
    sqrt: bool,
) -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, u32>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5: validate every geometry the launch derives bounds from. ---
    if n_features != n_features_t {
        return Err(PrimError::DimMismatch {
            dim: "cols",
            lhs: n_features,
            rhs: n_features_t,
        });
    }
    if n_query.checked_mul(n_features).map(|v| v != xq.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_query,
            cols: n_features,
            len: xq.len(),
        });
    }
    if n_train.checked_mul(n_features).map(|v| v != x_train.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_train,
            cols: n_features,
            len: x_train.len(),
        });
    }
    // `1 <= k <= min(n_train, FUSED_K_CAP)` — the k-vs-cols "shape mismatch"
    // shape mirrors `topk::validate_geometry`'s synthetic `"k"` operand.
    if k < 1 || k > n_train || k > FUSED_K_CAP {
        return Err(PrimError::ShapeMismatch {
            operand: "k",
            rows: 1,
            cols: k,
            len: n_train.min(FUSED_K_CAP),
        });
    }
    if n_query > FUSED_MAX_QUERY {
        return Err(PrimError::ShapeMismatch {
            operand: "n_query",
            rows: n_query,
            cols: 0,
            len: FUSED_MAX_QUERY,
        });
    }
    for (operand, dim) in [("rows", n_query), ("cols", n_train), ("features", n_features)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }

    let out_len = n_query * k;
    let client = pool.client().clone();

    let cy = ((n_query + FUSED_QUERY_BLOCK - 1) / FUSED_QUERY_BLOCK).max(1) as u32;
    let dim = CubeDim { x: 16, y: 16, z: 1 };

    // --- Train-dimension strip split (KNN-02 third pass). ---
    // A single strip launches only ceil(n_query/64) cubes, which quantizes
    // onto the SM count in 1-2 ragged waves at moderate n_query (measured on
    // the T4: the 157-cube config ran at ~60% of the 313-cube config's
    // per-pair rate). Strips multiply the cube count; each strip emits its own
    // k candidates and a short indexed merge selects the final k. Every strip
    // must hold >= 64 columns so its candidate union always covers k <= 16.
    let (strips, strip_len) = pick_strips(n_query, n_train, cy as usize);
    let cand_cols = strips * k;

    let count = CubeCount::Static(strips as u32, cy, 1);

    // Candidate buffers double as the FINAL buffers when strips == 1 (the
    // kernel then writes plain `rows × k` and no merge runs).
    let cand_len = n_query * cand_cols;
    let cval_handle = pool.acquire(cand_len * size_of::<F>());
    let cidx_handle = pool.acquire(cand_len * size_of::<u32>());

    // SAFETY: lengths are the carried/validated element counts; the kernel
    // bounds-checks every row/column against the scalar bounds.
    let x_arg = unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), xq.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(x_train.handle().clone(), x_train.len()) };
    let cval_arg = unsafe { ArrayArg::from_raw_parts(cval_handle.clone(), cand_len) };
    let cidx_arg = unsafe { ArrayArg::from_raw_parts(cidx_handle.clone(), cand_len) };

    if n_features <= 32 && dot_fused_selected() {
        // DOT-EXPANSION variant (KNN-02 fifth pass): `d² = ‖x‖² + ‖y‖² −
        // 2·x·y` with the norms precomputed here — the inner loop is pure
        // FMA, and the whole query tile is staged once per cube (cols fit a
        // single feature slice). sklearn's brute path uses this same
        // expansion; the clamp in-kernel keeps cancellation non-negative.
        // `MLRS_KNN_DIRECT=1` forces the cancellation-free direct kernel.
        // Norms via the dedicated one-launch GATHER kernel — NEVER
        // `prims::reduce::row_reduce`, whose per-row host round-trip loop is a
        // measured >100× pathology at these row counts (see the kernel docs).
        let xnorm_handle = pool.acquire(n_query * size_of::<F>());
        let ynorm_handle = pool.acquire(n_train * size_of::<F>());
        {
            let (nc, nd) = launch_dims_1d(n_query);
            let in_arg = unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), xq.len()) };
            let out_arg = unsafe { ArrayArg::from_raw_parts(xnorm_handle.clone(), n_query) };
            row_sumsq::launch::<F, ActiveRuntime>(
                &client, nc, nd, in_arg, out_arg, n_query as u32, n_features as u32,
            );
            let (nc, nd) = launch_dims_1d(n_train);
            let in_arg = unsafe { ArrayArg::from_raw_parts(x_train.handle().clone(), x_train.len()) };
            let out_arg = unsafe { ArrayArg::from_raw_parts(ynorm_handle.clone(), n_train) };
            row_sumsq::launch::<F, ActiveRuntime>(
                &client, nc, nd, in_arg, out_arg, n_train as u32, n_features as u32,
            );
        }
        let xn_arg = unsafe { ArrayArg::from_raw_parts(xnorm_handle.clone(), n_query) };
        let yn_arg = unsafe { ArrayArg::from_raw_parts(ynorm_handle.clone(), n_train) };
        let variant = dot_variant_selected(n_features);
        if variant == DotVariant::Vec8 {
            // KNN-03 r3: 4×8 register tile — 128-unit cubes, so the CubeDim
            // must match the kernel's unit map.
            let dim8 = CubeDim { x: 8, y: 16, z: 1 };
            euclidean_topk_fused_dot_vec8::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim8,
                x_arg,
                y_arg,
                xn_arg,
                yn_arg,
                cval_arg,
                cidx_arg,
                n_query as u32,
                n_train as u32,
                n_features as u32,
                k as u32,
                strip_len as u32,
                strips as u32,
            );
        } else if variant == DotVariant::Vec4 {
            // KNN-03: identical arithmetic per pair, vectorized tile accesses;
            // bitwise-gated against the scalar dot kernel in knn_test.rs.
            euclidean_topk_fused_dot_vec::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                x_arg,
                y_arg,
                xn_arg,
                yn_arg,
                cval_arg,
                cidx_arg,
                n_query as u32,
                n_train as u32,
                n_features as u32,
                k as u32,
                strip_len as u32,
                strips as u32,
            );
        } else {
            euclidean_topk_fused_dot::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                x_arg,
                y_arg,
                xn_arg,
                yn_arg,
                cval_arg,
                cidx_arg,
                n_query as u32,
                n_train as u32,
                n_features as u32,
                k as u32,
                strip_len as u32,
                strips as u32,
            );
        }
        // Both norm vectors are consumed by the launch above (same-stream
        // ordering protects any later reuse of the released buffers).
        DeviceArray::<ActiveRuntime, F>::from_raw(xnorm_handle, n_query).release_into(pool);
        DeviceArray::<ActiveRuntime, F>::from_raw(ynorm_handle, n_train).release_into(pool);
    } else {
        euclidean_topk_fused::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            x_arg,
            y_arg,
            cval_arg,
            cidx_arg,
            n_query as u32,
            n_train as u32,
            n_features as u32,
            k as u32,
            strip_len as u32,
            strips as u32,
        );
    }

    let (val_handle, idx_handle) = if strips == 1 {
        (cval_handle, cidx_handle)
    } else {
        // Indexed one-pass merge over the strips*k candidates per row. The
        // candidate POSITION tie-break equals the global lowest-index rule
        // (strips are contiguous ascending index ranges; see the kernel docs),
        // and the emitted index is the mapped global train index.
        let val_handle = pool.acquire(out_len * size_of::<F>());
        let idx_handle = pool.acquire(out_len * size_of::<u32>());
        let (mcount, mdim) = merge_launch_dims(n_query, cand_cols);
        let mcval_arg = unsafe { ArrayArg::from_raw_parts(cval_handle.clone(), cand_len) };
        let mcidx_arg = unsafe { ArrayArg::from_raw_parts(cidx_handle.clone(), cand_len) };
        let mval_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        let midx_arg = unsafe { ArrayArg::from_raw_parts(idx_handle.clone(), out_len) };
        select_k_onepass_indexed::launch::<F, ActiveRuntime>(
            &client,
            mcount,
            mdim,
            mcval_arg,
            mcidx_arg,
            mval_arg,
            midx_arg,
            n_query as u32,
            cand_cols as u32,
            k as u32,
        );
        // The candidate buffers are consumed by the merge launch above and
        // never read again (same-stream ordering protects any later reuse).
        DeviceArray::<ActiveRuntime, F>::from_raw(cval_handle, cand_len).release_into(pool);
        DeviceArray::<ActiveRuntime, u32>::from_raw(cidx_handle, cand_len).release_into(pool);
        (val_handle, idx_handle)
    };

    // Euclidean sqrt over ONLY the returned k values per row (D-08), identical
    // to `top_k`'s boundary pass.
    if sqrt {
        let (scount, sdim) = launch_dims_1d(out_len);
        let in_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        let sout_arg = unsafe { ArrayArg::from_raw_parts(val_handle.clone(), out_len) };
        sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
    }

    Ok((
        DeviceArray::from_raw(val_handle, out_len),
        DeviceArray::from_raw(idx_handle, out_len),
    ))
}

/// Choose the train-dimension strip count + per-strip length (in columns) for
/// the fused kernel.
///
/// Targets at least [`FUSED_TARGET_CUBES`] total cubes (`query_blocks ×
/// strips`) so the launch is several waves deep on any current GPU, while
/// keeping every strip a multiple of the 64-wide block with the LAST strip
/// >= 64 columns (so every strip's candidate union covers `k <=
/// FUSED_K_CAP <= 64`). Returns `(1, n_train_rounded)` when splitting is not
/// possible or not useful. `MLRS_KNN_STRIPS` overrides the strip count for
/// on-target A/B.
fn pick_strips(_n_query: usize, n_train: usize, query_blocks: usize) -> (usize, usize) {
    let forced = std::env::var("MLRS_KNN_STRIPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    // DEFAULT 1 (measured on the T4, KNN-02d): strip counts of 4-8 REGRESSED
    // every ladder config by 25-60% — the k-round merge phase is a per-cube
    // fixed cost, and strips multiply it faster than they amortize the launch
    // tail. The machinery is kept behind `MLRS_KNN_STRIPS` for future
    // on-target A/B (e.g. very small n_query with very large n_train).
    let desired = match forced {
        Some(s) => s.clamp(1, 64),
        None => 1,
    };
    let _ = query_blocks;
    // Round the per-strip length UP to the 64-wide block, then drop strips
    // whose start would fall past the training set; keep the last strip at
    // >= 64 columns by shrinking the count until it fits.
    let mut strips = desired.min(n_train / 64).max(1);
    loop {
        let strip_len = n_train.div_ceil(strips).div_ceil(64) * 64;
        let used = (strips - 1) * strip_len;
        if strips == 1 || (used < n_train && n_train - used >= 64) {
            return (strips, strip_len);
        }
        strips -= 1;
    }
}

/// Total-cube target for [`pick_strips`]: comfortably more than the resident
/// capacity of current GPUs (a T4 holds 40 SMs × 2-4 cubes), so the tail wave
/// is a small fraction of the launch.
#[allow(dead_code)]
const FUSED_TARGET_CUBES: usize = 512;

/// Launch config for the indexed candidate merge: one cube per query row, a
/// POWER-OF-TWO width <= 256 covering the (short) candidate row.
fn merge_launch_dims(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let capped = cols.min(256) as u32;
    let width = 1u32 << (31 - capped.max(1).leading_zeros());
    (
        CubeCount::Static((rows as u32).max(1), 1, 1),
        CubeDim { x: width, y: 1, z: 1 },
    )
}
