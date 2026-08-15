//! `kernel_matrix` — pairwise kernel-matrix primitive (PRIM-08).
//!
//! Computes the general kernel matrix `K(X, Y)` (an `rows_x × rows_y` matrix,
//! D-02) for every non-`precomputed` kernel family sklearn's `pairwise_kernels`
//! ships, which is the full string surface `KernelRidge(kernel=…)` accepts
//! (D-01):
//!   - `linear`:  `K = X·Yᵀ` (the GEMM base op, identity map).
//!   - `rbf`:     `K = exp(-γ·‖xᵢ − yⱼ‖²)` (squared-euclidean distance base op,
//!                then `exp(-γ··)` map).
//!   - `poly`:    `K = (γ·⟨xᵢ, yⱼ⟩ + coef0)^degree` (GEMM base op, then powf map).
//!   - `sigmoid`: `K = tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)` (GEMM base op, then tanh map).
//!   - `laplacian`: `K = exp(-γ·‖xᵢ − yⱼ‖₁)` (L1 pairwise base op, then the same
//!                `exp(-γ··)` map as `rbf`).
//!   - `cosine`:  `K = ⟨x̂ᵢ, ŷⱼ⟩` (GEMM over L2-normalised rows, identity map).
//!   - `additive_chi2`: `K = -Σₖ (xᵢₖ − yⱼₖ)²/(xᵢₖ + yⱼₖ)` (a dedicated pairwise
//!                base op, identity map; non-negative operands required).
//!   - `chi2`:    `K = exp(γ·additive_chi2)` (the same base op, then the `rbf`
//!                map with a NEGATED γ, which is exactly `exp(γ·A)`).
//!
//! `precomputed` is deliberately NOT a variant here: it names the absence of a
//! kernel computation, and the estimator that has the caller's `K` in hand
//! simply never calls this prim.
//!
//! ## Three base ops, eight kernels
//! Only `additive_chi2` needed a new base op. `laplacian` and `cosine` reach
//! their results by re-pointing existing ones (the L1 arm of `metric_distance`,
//! and GEMM over normalised rows), and `chi2` reaches its by re-signing an
//! existing map's argument. That is the point of the base-op → map factoring:
//! the number of transcendental evaluations in this file did not grow when the
//! kernel count doubled.
//!
//! ## Composition (the covariance.rs base-op → in-place-map idiom)
//! Like [`crate::prims::covariance`], `kernel_matrix` is a thin host
//! orchestration over already-validated v1 base ops plus one SharedMemory-free
//! per-element map kernel (added in `mlrs-kernels` by the Wave-1 plan):
//!   - `linear`/`poly`/`sigmoid` use [`crate::prims::gemm::gemm`] (`transb =
//!     true`, the `X·Yᵀ` base) as the base op (`gemm.rs:54`).
//!   - `rbf` uses [`crate::prims::distance::distance`] (`sqrt = false`, the
//!     squared-euclidean base, `distance.rs:79`).
//! The per-kernel map then runs IN PLACE over the base buffer (input handle ==
//! output handle), exactly the covariance `scale`-in-place idiom
//! (`covariance.rs:190-204`); the result IS the base buffer, mapped in place
//! (D-02/D-03 single code path). `linear` is the identity map — it skips the map
//! launch and returns the GEMM buffer directly.
//!
//! ## Validate-before-launch (ASVS V5 / T-08-01-01)
//! The geometry guard (`rows_x·cols == x.len()`, `rows_y·cols == y.len()`,
//! reject empty geometry, `out` len == `rows_x·rows_y`) runs BEFORE any `unsafe`
//! kernel launch, returning a typed [`PrimError`], never an out-of-bounds device
//! read — the same contract as `covariance.rs:212-262` / `gemm.rs`.
//!
//! ## Wave-0 scaffold status
//! This file is the 08-01 Wave-0 COMPILING STUB: it defines the public surface
//! (`Kernel<F>` enum + the `kernel_matrix` host-fn signature) that the Wave-0
//! test scaffold compiles against, with the geometry validation REAL (so the
//! signature and error type are real) but the compute path left as `todo!()` for
//! the Wave-1 plan (08-02) to fill (it adds the `mlrs-kernels` map kernel + the
//! base-op dispatch). Do NOT write the map kernel here — that is Wave 1.
//!
//! Tests live in `crates/mlrs-backend/tests/kernel_matrix_test.rs` (AGENTS.md §2
//! — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use mlrs_kernels::{additive_chi2_dist, poly_map, rbf_map, sigmoid_map};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::distance::{distance, metric_distance};
use crate::prims::gemm::gemm;
use crate::prims::knn_graph::Metric;
use crate::runtime::ActiveRuntime;

/// The typed kernel-family selector (D-01) `kernel_matrix` matches on to pick the
/// base op + per-element map. Generic over the float element type `F` so the
/// kernel hyperparameters (`gamma` / `degree` / `coef0`) carry the same precision
/// as the data, matching the `<F: Float + CubeElement + Pod>` bound the prim
/// functions use (`covariance.rs`).
///
/// `degree` is stored as `F` (not an integer) because sklearn's poly kernel takes
/// a real degree (`Interval(Real, 1, None)`) and the map uses `F::powf` — the
/// sklearn-faithful real-exponent form (RESEARCH §kernel_matrix.rs / Pitfall 7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel<F>
where
    F: Float + CubeElement + Pod,
{
    /// Linear kernel `K = X·Yᵀ` — the GEMM base op, identity map (no map launch).
    Linear,
    /// RBF (Gaussian) kernel `K = exp(-γ·‖xᵢ − yⱼ‖²)` — squared-euclidean
    /// distance base op, then the `exp(-γ··)` map.
    Rbf {
        /// Kernel coefficient `γ` (resolved to `1/n_features` by the caller when
        /// sklearn's `gamma=None` default is requested — D-05).
        gamma: F,
    },
    /// Polynomial kernel `K = (γ·⟨xᵢ, yⱼ⟩ + coef0)^degree` — GEMM base op, then
    /// the `powf(γ·g + coef0, degree)` map.
    Poly {
        /// Kernel coefficient `γ`.
        gamma: F,
        /// Polynomial degree (real, `≥ 1` — validated by the estimator).
        degree: F,
        /// Independent term `coef0`.
        coef0: F,
    },
    /// Sigmoid kernel `K = tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)` — GEMM base op, then the
    /// `tanh(γ·g + coef0)` map.
    Sigmoid {
        /// Kernel coefficient `γ`.
        gamma: F,
        /// Independent term `coef0`.
        coef0: F,
    },
    /// Laplacian kernel `K = exp(-γ·‖xᵢ − yⱼ‖₁)` — the L1 (Manhattan) pairwise
    /// distance base op, then the SAME `exp(-γ··)` map RBF uses.
    ///
    /// The only thing separating this from `Rbf` is which norm feeds the
    /// exponential, which is why it reuses `rbf_map` rather than getting a map
    /// of its own: `laplacian_kernel` in sklearn is `rbf_kernel` with
    /// `manhattan_distances` substituted for the squared Euclidean, and keeping
    /// one map means the two can never drift in how they evaluate `exp`.
    Laplacian {
        /// Kernel coefficient `γ` (resolved to `1/n_features` by the caller when
        /// sklearn's `gamma=None` default is requested — D-05).
        gamma: F,
    },
    /// Cosine kernel `K = ⟨x̂ᵢ, ŷⱼ⟩` over L2-normalised rows — sklearn's
    /// `cosine_similarity`. Parameterless (`gamma`/`degree`/`coef0` do not apply).
    Cosine,
    /// Exponential chi-squared kernel `K = exp(γ·A)` where `A` is the
    /// [`Kernel::AdditiveChi2`] value (already negative), i.e.
    /// `exp(-γ·Σₖ (xᵢₖ − yⱼₖ)²/(xᵢₖ + yⱼₖ))`.
    ///
    /// Requires a NON-NEGATIVE `x`/`y` (checked by the caller, as sklearn's
    /// `check_non_negative` does) and an EXPLICIT `γ` — see
    /// `KernelRidge::fit` for why `gamma = None` is an error here and a
    /// `1/n_features` default everywhere else.
    Chi2 {
        /// Kernel coefficient `γ`.
        gamma: F,
    },
    /// Additive chi-squared kernel `K = -Σₖ (xᵢₖ − yⱼₖ)²/(xᵢₖ + yⱼₖ)` — sklearn's
    /// `additive_chi2_kernel`. Parameterless, non-negative operands required.
    ///
    /// Note this kernel is NOT positive definite in the strict sense sklearn's
    /// docs warn about; a `KernelRidge` fit on it can legitimately hit a
    /// non-SPD `(K + αI)` for a small `α`, which surfaces as
    /// `PrimError::NotPositiveDefinite` rather than a silently wrong solve.
    AdditiveChi2,
}

/// Compute the general kernel matrix `K(X, Y)` (D-02): an `rows_x × rows_y`
/// row-major matrix whose `(i, j)` entry is the chosen [`Kernel`] applied to the
/// `i`-th row of `X` and the `j`-th row of `Y`. Both operands are row-major
/// `(rows, cols)` device buffers sharing the feature dimension `cols`.
///
/// - `x` is the `rows_x × cols` left operand; `y` is the `rows_y × cols` right
///   operand (for the symmetric training Gram `K(X, X)` the caller passes `y =
///   x`, D-02).
/// - Geometry is validated against the operand element counts **before** any
///   launch (`rows_x·cols == x.len()`, `rows_y·cols == y.len()`, non-empty,
///   `out` len == `rows_x·rows_y`); a mismatch returns
///   [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`] (ASVS V5 /
///   T-08-01-01).
/// - The `rows_x × rows_y` output is acquired from `pool` when `out` is `None`,
///   else the supplied buffer is reused (D-11). The result stays device-resident
///   (D-05) — NO host round-trip inside this API.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
///
/// **Wave-0 stub:** geometry validation is real; the compute path is `todo!()`
/// pending the Wave-1 plan (08-02), which adds the `mlrs-kernels` map kernel and
/// the base-op (`gemm`/`distance`) dispatch.
#[allow(clippy::too_many_arguments)]
pub fn kernel_matrix<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    kernel: Kernel<F>,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- Geometry guard (ASVS V5 / T-08-01-01): validate BEFORE any unsafe
    //     launch so a wrong shape is a recoverable typed error, not an
    //     out-of-bounds device read. Mirrors covariance.rs:212-262 / gemm.rs. ---
    validate_geometry(
        x.len(),
        (rows_x, cols),
        y.len(),
        (rows_y, cols_y),
        out.as_ref().map(DeviceArray::len),
    )?;

    // --- Base-op dispatch + in-place map (the covariance.rs:151-204 idiom).
    //       linear  → gemm(x, y, transb=true), return directly (identity map).
    //       rbf     → distance(x, y, sqrt=false), then exp(-gamma··) map in place.
    //       poly    → gemm(x, y, transb=true), then powf(gamma·g+coef0, degree).
    //       sigmoid → gemm(x, y, transb=true), then tanh(gamma·g+coef0).
    //     The map runs IN PLACE over the base buffer (covariance.rs:190-204) —
    //     the result IS the base buffer, mapped in place (D-02/D-03). The full
    //     general rows_x × rows_y K(X, Y) is always computed (D-02 — no symmetry
    //     special-case). The n×n operand stays in GLOBAL memory (no SharedMemory
    //     tile; gfx1100 LDS ≤ 65536 B — T-08-02-02). ---
    match kernel {
        // Linear: K = X·Yᵀ. The GEMM buffer IS the kernel matrix (identity map),
        // so we return it directly with NO map launch.
        Kernel::Linear => {
            // logical lhs (m, k) = (rows_x, cols); rhs (k, n) = (cols, rows_y),
            // transb=true ⇒ stored y is (rows_y, cols) = `y`'s layout.
            gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)
        }
        // RBF: squared-euclidean base (sqrt=false ⇒ ‖xᵢ − yⱼ‖²), then the
        // exp(-γ··) map in place over that base buffer (D-03 / Pitfall 4).
        Kernel::Rbf { gamma } => {
            let base = distance::<F>(pool, x, (rows_x, cols), y, (rows_y, cols_y), false, out)?;
            let n = rows_x * rows_y;
            if host_map_applicable::<F>() {
                let g = host_to_f64(gamma);
                return Ok(map_in_place_host::<F>(pool, base, n, |d2| (-g * d2).exp()));
            }
            launch_map_in_place(pool, &base, n, |client, count, dim, in_arg, out_arg| {
                rbf_map::launch::<F, ActiveRuntime>(client, count, dim, in_arg, out_arg, gamma);
            });
            Ok(base)
        }
        // Poly: XYᵀ Gram base, then powf(γ·g + coef0, degree) map in place.
        Kernel::Poly { gamma, degree, coef0 } => {
            let base =
                gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)?;
            let n = rows_x * rows_y;
            if host_map_applicable::<F>() {
                let (g, c0, deg) = (host_to_f64(gamma), host_to_f64(coef0), host_to_f64(degree));
                return Ok(map_in_place_host::<F>(pool, base, n, |v| (g * v + c0).powf(deg)));
            }
            launch_map_in_place(pool, &base, n, |client, count, dim, in_arg, out_arg| {
                poly_map::launch::<F, ActiveRuntime>(
                    client, count, dim, in_arg, out_arg, gamma, coef0, degree,
                );
            });
            Ok(base)
        }
        // Sigmoid: XYᵀ Gram base, then tanh(γ·g + coef0) map in place.
        Kernel::Sigmoid { gamma, coef0 } => {
            let base =
                gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)?;
            let n = rows_x * rows_y;
            if host_map_applicable::<F>() {
                let (g, c0) = (host_to_f64(gamma), host_to_f64(coef0));
                return Ok(map_in_place_host::<F>(pool, base, n, |v| (g * v + c0).tanh()));
            }
            launch_map_in_place(pool, &base, n, |client, count, dim, in_arg, out_arg| {
                sigmoid_map::launch::<F, ActiveRuntime>(
                    client, count, dim, in_arg, out_arg, gamma, coef0,
                );
            });
            Ok(base)
        }
        // Laplacian: L1 (Manhattan) pairwise base, then the SAME exp(-γ··) map
        // RBF uses. `metric_distance` returns the TRUE L1 distance (its
        // needs-boundary-sqrt flag is false for Manhattan), so the base is fed to
        // the map unmodified — the flag is discarded deliberately, not dropped by
        // oversight, and the debug assertion below pins that.
        Kernel::Laplacian { gamma } => {
            let (base, needs_sqrt) = metric_distance::<F>(
                pool,
                x,
                (rows_x, cols),
                y,
                (rows_y, cols_y),
                Metric::Manhattan,
                out,
            )?;
            debug_assert!(
                !needs_sqrt,
                "Manhattan returns the true L1 distance; a boundary sqrt would \
                 mean the metric dispatch changed under this kernel"
            );
            let n = rows_x * rows_y;
            if host_map_applicable::<F>() {
                let g = host_to_f64(gamma);
                return Ok(map_in_place_host::<F>(pool, base, n, |d1| (-g * d1).exp()));
            }
            launch_map_in_place(pool, &base, n, |client, count, dim, in_arg, out_arg| {
                rbf_map::launch::<F, ActiveRuntime>(client, count, dim, in_arg, out_arg, gamma);
            });
            Ok(base)
        }
        // Cosine: the LINEAR kernel over L2-normalised rows — sklearn's
        // `cosine_similarity` is `normalize(X) @ normalize(Y).T` and nothing
        // else, so this is the linear arm with normalised operands and no map.
        //
        // The normalisation runs on the HOST (the `knn_graph` cosine precedent):
        // it is an O(n·d) pass in front of the O(n²·d) GEMM, and doing it here
        // rather than in a kernel is what lets the zero-row rule be sklearn's
        // exactly — `normalize` divides by 1 where the norm is 0, leaving the row
        // zero, and a device reciprocal would have to reproduce that guard.
        Kernel::Cosine => {
            let xn = l2_normalized_copy::<F>(pool, x, rows_x, cols);
            // `y` may BE `x` (the symmetric training Gram passes `y = x`), so the
            // second operand is normalised into its own buffer unconditionally
            // rather than aliased — the buffers are released independently below.
            let yn = l2_normalized_copy::<F>(pool, y, rows_y, cols_y);
            let k = gemm::<F>(pool, &xn, (rows_x, cols), &yn, (cols, rows_y), false, true, out);
            xn.release_into(pool);
            yn.release_into(pool);
            k
        }
        // Chi2: the additive-chi² base A (already negative), then exp(γ·A).
        // `rbf_map` computes `exp(-γ'·v)`, so passing `γ' = -γ` gives exactly
        // `exp(γ·A)` with no second map kernel — the sign lives in the argument,
        // which is why `Chi2` needs no transcendental of its own.
        Kernel::Chi2 { gamma } => {
            let base =
                additive_chi2_base::<F>(pool, x, (rows_x, cols), y, (rows_y, cols_y), out)?;
            let n = rows_x * rows_y;
            if host_map_applicable::<F>() {
                let g = host_to_f64(gamma);
                return Ok(map_in_place_host::<F>(pool, base, n, |a| (g * a).exp()));
            }
            let neg_gamma = f64_to_host::<F>(-host_to_f64(gamma));
            launch_map_in_place(pool, &base, n, |client, count, dim, in_arg, out_arg| {
                rbf_map::launch::<F, ActiveRuntime>(client, count, dim, in_arg, out_arg, neg_gamma);
            });
            Ok(base)
        }
        // AdditiveChi2: the base IS the kernel (identity map, like `linear`).
        Kernel::AdditiveChi2 => {
            additive_chi2_base::<F>(pool, x, (rows_x, cols), y, (rows_y, cols_y), out)
        }
    }
}

/// An L2-row-normalised device copy of `m` (`rows × cols`), for the cosine
/// kernel. A zero row stays zero (sklearn's `normalize` divides such a row by 1),
/// which is what makes `cosine_similarity` return 0 rather than NaN against it.
///
/// Returns a NEW buffer; the caller owns it and must release it. `m` is left
/// untouched — it is usually fitted state (`X_fit_`) or the caller's borrowed
/// input, neither of which this prim may mutate.
fn l2_normalized_copy<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    m: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let host: Vec<F> = m.to_host(pool);
    let mut out: Vec<F> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        // Accumulate the norm in f64 regardless of `F`: at f32 a row of large
        // features can overflow the sum of squares long before the row itself is
        // anywhere near the f32 range, and the resulting `inf` norm would zero a
        // perfectly ordinary row.
        let norm = row
            .iter()
            .map(|&v| host_to_f64(v).powi(2))
            .sum::<f64>()
            .sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for &v in row {
            out.push(f64_to_host::<F>(host_to_f64(v) * inv));
        }
    }
    DeviceArray::from_host(pool, &out)
}

/// The additive chi-squared base `A[i][j] = -Σₖ (xᵢₖ − yⱼₖ)²/(xᵢₖ + yⱼₖ)`
/// (`rows_x × rows_y`, row-major), shared by [`Kernel::AdditiveChi2`] (which
/// returns it as-is) and [`Kernel::Chi2`] (which exponentiates it).
///
/// Non-negative operands are a CALLER obligation, validated host-side by the
/// estimator the way sklearn's `check_non_negative` does — the kernel's
/// `nom > 0` term guard assumes it.
fn additive_chi2_base<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, _cols_y): (usize, usize),
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // WR-03 (the `distance_direct` precedent): the three dims are cast to u32 for
    // the launch, so reject an overflowing dimension BEFORE the launch rather
    // than let the cast truncate into a bad loop bound.
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
    let out_len = rows_x.checked_mul(rows_y).ok_or(PrimError::Overflow {
        operand: "additive_chi2",
        lhs: rows_x,
        rhs: rows_y,
    })?;
    let out_handle = match &out {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * std::mem::size_of::<F>()),
    };
    let client = pool.client().clone();

    // SAFETY: the operand lengths are the geometry-validated element counts the
    // caller checked in `validate_geometry`, and the kernel bounds-checks
    // `i < rows_x && j < rows_y` before any index.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

    let (count, dim) = launch_dims_2d(rows_x, rows_y);
    additive_chi2_dist::launch::<F, ActiveRuntime>(
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
    Ok(DeviceArray::from_raw(out_handle, out_len))
}

/// 2D per-output-element launch config for [`additive_chi2_base`], the same
/// 16×16 cube the direct pairwise distance kernels use (`distance.rs`).
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

/// Must the per-element kernel map run on the HOST rather than as a device
/// launch?
///
/// True exactly when the element type is `f64` and the backend cannot evaluate
/// f64 transcendentals (`capability::f64_transcendental_supported`). On such a
/// backend the `rbf`/`poly`/`sigmoid` map kernels — `exp`, `powf`, `tanh` — do
/// not fail at launch: the driver's shader compiler either SEGFAULTS or emits
/// garbage, which is how a wgpu f64 `KernelRidge` fit ended up handing the
/// Cholesky solve a matrix with a `-5e204` pivot and how `SpectralClustering`'s
/// affinity produced an `eig` residual of `6.5e63`.
fn host_map_applicable<F>() -> bool {
    std::mem::size_of::<F>() == 8 && !crate::capability::f64_transcendental_supported()
}

/// Apply a per-element map on the HOST, in `f64`, over the base buffer.
///
/// The device twin of this is [`launch_map_in_place`]; both rewrite the base
/// buffer in place, so the caller is unchanged. Only the MAP moves — the base op
/// (`gemm` / `distance`, the `O(rows_x·rows_y·cols)` work) still runs on device,
/// and what crosses the bus is the `rows_x · rows_y` result that the map would
/// have rewritten anyway. The host pass is therefore `O(n²)` against the base
/// op's `O(n²·d)`, not a fallback to computing the kernel matrix on the CPU.
///
/// `f64` throughout, so on the backends that take this path the map is evaluated
/// at FULL precision — the same arithmetic the device kernel performs on a
/// backend that supports it.
fn map_in_place_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    base: DeviceArray<ActiveRuntime, F>,
    n: usize,
    f: impl Fn(f64) -> f64,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let mut host: Vec<F> = base.to_host(pool);
    for v in host.iter_mut().take(n) {
        *v = f64_to_host::<F>(f(host_to_f64(*v)));
    }
    // cubecl 0.10 has no in-place write into an existing handle (see
    // `DeviceArray::from_host`), so the base buffer is RELEASED first and the
    // mapped values re-staged — `from_host` then recycles the just-freed
    // byte-size off the pool free-list, so no second live `rows_x · rows_y`
    // allocation exists at any point. This is the same release-then-restage
    // idiom `ridge.rs` uses for its `α`-on-the-diagonal Gram rewrite.
    base.release_into(pool);
    DeviceArray::from_host(pool, &host)
}

/// Launch a per-element map IN PLACE over `base` (input handle == output handle),
/// the covariance.rs:190-204 scale-in-place idiom. `n` is the element count
/// (`rows_x · rows_y`); the closure receives the client + launch dims + the
/// in/out `ArrayArg`s (both wrapping the SAME `base` handle) so the map rewrites
/// the base buffer with no parallel allocation (T-08-02-02 — the in-place map
/// reuses the base buffer).
fn launch_map_in_place<F, L>(
    pool: &mut BufferPool<ActiveRuntime>,
    base: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    launch: L,
) where
    F: Float + CubeElement + Pod,
    L: FnOnce(
        &cubecl::client::ComputeClient<ActiveRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<ActiveRuntime>,
        ArrayArg<ActiveRuntime>,
    ),
{
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: `n` is the carried base-op output element count (rows_x · rows_y,
    // itself derived from the validated geometry); each map kernel bounds-checks
    // `tid < input.len()` (T-08-02-01). input and output are the SAME handle so
    // the map is applied in place over the reused base buffer (no parallel
    // allocation — T-08-02-02).
    let in_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    launch(&client, count, dim, in_arg, out_arg);
}

/// Standard ceiling-division 1D launch config for the in-place map pass (the
/// `elementwise` per-element launch idiom, copied from covariance.rs:266-273).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256usize;
    // Compute the cube count in `usize` and check the `u32` launch-grid cast
    // (WR-02): an unchecked `n as u32` silently wraps for `n > u32::MAX`,
    // under-provisioning threads so trailing elements are never mapped — a silent
    // wrong-result. The kernel-matrix problem sizes are small today, but the guard
    // turns the overflow into a loud panic instead.
    let cubes = u32::try_from((n + block - 1) / block)
        .expect("element count exceeds u32 launch-grid limit");
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim {
            x: block as u32,
            y: 1,
            z: 1,
        },
    )
}

/// Validate the kernel-matrix operand geometry (ASVS V5 / T-08-01-01). `x` is
/// `rows_x × cols`, `y` is `rows_y × cols` (the SHARED feature dimension must
/// agree); the output (if supplied) must be the `rows_x × rows_y` kernel matrix.
/// Empty geometry is rejected at the boundary (a 0-row / 0-col operand has no
/// well-defined kernel matrix).
fn validate_geometry(
    x_len: usize,
    (rows_x, cols): (usize, usize),
    y_len: usize,
    (rows_y, cols_y): (usize, usize),
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    // x must be a well-formed rows_x × cols.
    if rows_x
        .checked_mul(cols)
        .map(|v| v != x_len)
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    // y must be a well-formed rows_y × cols_y.
    if rows_y
        .checked_mul(cols_y)
        .map(|v| v != y_len)
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: rows_y,
            cols: cols_y,
            len: y_len,
        });
    }
    // The shared feature dimension must agree (K(X, Y) is only defined when X and
    // Y live in the same feature space).
    if cols != cols_y {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: cols,
            rhs: cols_y,
        });
    }
    // Reject empty geometry at the boundary (no well-defined kernel matrix).
    if rows_x == 0 || rows_y == 0 || cols == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    // The output (if supplied) must be the rows_x × rows_y kernel matrix.
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
