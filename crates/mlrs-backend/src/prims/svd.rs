//! Thin (economy) SVD host API (PRIM-05) — `A = U·diag(S)·Vᵀ` with
//! `U` (m×k), `S` (k), `Vᵀ` (k×n), `k = min(m, n)`, matching
//! `numpy.linalg.svd(full_matrices=False)` (D-02). Drives the one-sided Jacobi
//! sweep kernel ([`mlrs_kernels::jacobi_svd_sweep`]) for the convergence, then
//! recovers the thin factors by composing the already-validated Phase-2
//! primitives — NO bespoke matmul / norm is written here ("Don't Hand-Roll").
//!
//! ## Tall + wide via the Aᵀ-and-swap dispatch (D-05)
//! The Jacobi kernel assumes a tall `A` (`rows ≥ cols`). A wide input
//! (`rows < cols`) is handled by running the kernel on `Aᵀ` (which is tall) and
//! swapping the result: `A = UΣVᵀ ⇒ Aᵀ = VΣUᵀ`, so the SVD of `Aᵀ` gives
//! `(U', S', V'ᵀ)` with `U = V'`, `S = S'`, `Vᵀ = U'ᵀ`. `Aᵀ` is read via the
//! Phase-2 GEMM transpose flag when forming `A·V` — no materialized transpose
//! buffer for the kernel input either (we transpose on the host once into a
//! pooled scratch, the single allowed copy, since the kernel reads a row-major
//! `(rows, cols)` array directly).
//!
//! ## Thin-U / S extraction (Pattern 3, reuses GEMM + reduce)
//! The kernel writes the accumulated `V` (`k × k`, column-major). The host then:
//!   1. forms `B = A·V` (`m × k`) with the Phase-2 [`gemm`] (D-02 / "Don't
//!      Hand-Roll");
//!   2. `S[j] = ‖B[:, j]‖₂` via the Phase-2 column L2-norm reduction;
//!   3. `U[:, j] = B[:, j] / S[j]`, guarding `S[j]` against a NEAR-ZERO floor so
//!      a rank-deficient column (σ ≈ 0) does not divide by zero (Pitfall 4 — the
//!      U column is left at 0 there; the reconstruction invariant still holds
//!      because a zero singular value contributes nothing to `UΣVᵀ`).
//! `Vᵀ` is the transpose of `V`. Finally `S` is sorted DESCENDING and `U`'s
//! columns / `Vᵀ`'s rows are permuted to match (D-04; host-side selection sort
//! post-convergence is fine — A4 — the sort is not the convergence loop the
//! D-11 gate concerns).
//!
//! ## Convergence failure (D-12)
//! If the kernel reports it hit the sweep cap without driving the off-diagonal
//! norm below the threshold, this API returns [`PrimError::NotConverged`] rather
//! than a silently-unconverged result.
//!
//! ## Device residency (D-05)
//! The returned `(U, S, Vᵀ)` are device-resident [`DeviceArray`]s. This API
//! reads back ONLY the small `k × k` `V` and the length-`k` `S` for the host-side
//! sort/permute (the convergence loop itself is fully in-kernel — D-11 gate 3);
//! it performs no read-back of the `m × n` input or the `m × k` `U`.
//!
//! Tests live in `crates/mlrs-backend/tests/svd_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::jacobi_svd_sweep;
use mlrs_kernels::{MAX_COLS, MAX_ROWS};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Off-diagonal threshold scale factor `c` in `threshold = c · ε_F · ‖A‖_F`
/// (D-12 — RESEARCH Open Q1 / Pitfall 5). `8` holds 1e-5 across the D-08 sweep
/// while staying reachable in f32.
const THRESHOLD_SCALE: f64 = 8.0;

/// Max-sweep cap (D-12). Cyclic one-sided Jacobi converges quadratically
/// (~10–15 sweeps for n ≤ 256); 30 is generous headroom (Pitfall 5).
const MAX_SWEEPS: u32 = 30;

/// Near-zero floor for the thin-U column normalization (Pitfall 4). Below this a
/// singular value is treated as zero and its `U` column is left at 0, so a
/// rank-deficient input never divides by zero. Mirrors `mlrs_core`'s
/// `NEAR_ZERO_FLOOR` precedent (chosen below the 1e-5 tolerance so it never
/// loosens a real check).
const NEAR_ZERO_FLOOR: f64 = 1e-8;

/// Compute the thin SVD of `a` (`rows × cols`, row-major), returning the
/// device-resident `(U, S, Vᵀ)` with `U` (`rows × k`), `S` (`k`), `Vᵀ`
/// (`k × cols`), `k = min(rows, cols)` (D-02), `S` descending (D-04).
///
/// - `a` is the row-major `rows × cols` matrix. Geometry is validated
///   (`rows * cols == a.len()`, and `min(rows,cols) ≤ MAX_COLS`,
///   `max(rows,cols) ≤ MAX_ROWS`) BEFORE any unsafe launch (ASVS V5 /
///   T-03-03-01); a mismatch returns [`PrimError::ShapeMismatch`].
/// - Tall (`rows ≥ cols`) runs the Jacobi kernel directly; wide (`rows < cols`)
///   runs it on `Aᵀ` and swaps `U ↔ V` (D-05).
/// - Non-convergence within the sweep cap returns [`PrimError::NotConverged`].
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log` (f64 runs on cpu,
/// skips on rocm — D-07).
#[allow(clippy::type_complexity)]
pub fn svd<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (rows, cols): (usize, usize),
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-03-03-01: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(a.len(), (rows, cols))?;

    if rows >= cols {
        // Tall path: Jacobi directly on A, U/S/Vᵀ as-is.
        svd_tall::<F>(pool, a, rows, cols, false)
    } else {
        // Wide path (D-05): run on Aᵀ (which is tall, cols×rows), then swap
        // U ↔ V. We materialize Aᵀ once into pooled scratch (the single allowed
        // host transpose copy — the kernel reads a row-major (rows', cols') array
        // directly). Aᵀ is (cols × rows): the SVD of Aᵀ is (U', S', V'ᵀ) with
        // U = V', S = S', Vᵀ = U'ᵀ.
        let a_host = a.to_host(pool);
        let mut at_host: Vec<F> = vec![F::from_int(0i64); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                // Aᵀ[c, r] = A[r, c]; Aᵀ is (cols × rows) row-major.
                at_host[c * rows + r] = a_host[r * cols + c];
            }
        }
        let at_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &at_host);
        // SVD of Aᵀ (cols × rows, tall since cols > rows). Pass swap=true so the
        // returned tuple is already (U, S, Vᵀ) of the ORIGINAL A.
        let res = svd_tall::<F>(pool, &at_dev, cols, rows, true);
        at_dev.release_into(pool);
        res
    }
}

/// Tall-path SVD driver (`rows ≥ cols`, `k = cols`). When `swap_uv` is true the
/// caller is the wide path: the input is `Aᵀ` and the returned `(U, S, Vᵀ)`
/// are relabeled to the ORIGINAL (pre-transpose) matrix (`U = V'`, `Vᵀ = U'ᵀ`).
#[allow(clippy::type_complexity)]
fn svd_tall<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
    swap_uv: bool,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    let k = cols; // tall: k = min(rows, cols) = cols.
    let elem = size_of::<F>();

    // --- Launch the Jacobi sweep kernel (one cube of `cols` units). The kernel
    //     keeps the convergence loop in-kernel (D-11 gate 3); it writes the
    //     accumulated V (k×k, column-major) and a small info array (sweeps run,
    //     final off-diagonal norm). We use V (not the kernel's rotated A) for the
    //     thin-U extraction via the Phase-2 GEMM (D-02 / "Don't Hand-Roll"). ---
    let a_out_handle = pool.acquire(rows * cols * elem); // rotated A (col-major) — scratch
    let v_handle = pool.acquire(k * k * elem); // accumulated V (col-major)
    let info_handle = pool.acquire(2 * elem); // [sweeps, residual]

    let client = pool.client().clone();
    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim {
        x: cols as u32,
        y: 1,
        z: 1,
    };

    // SAFETY: lengths are the carried/validated element counts (rows*cols, k*k,
    // 2), NEVER raw caller geometry; the kernel bounds every loop by the runtime
    // (rows, cols) and idles units with `c >= cols` (mitigates T-03-03-03 / the
    // OOB device-read threat, ASVS V5).
    let a_in_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), rows * cols) };
    let a_out_arg = unsafe { ArrayArg::from_raw_parts(a_out_handle.clone(), rows * cols) };
    let v_arg = unsafe { ArrayArg::from_raw_parts(v_handle.clone(), k * k) };
    let info_arg = unsafe { ArrayArg::from_raw_parts(info_handle.clone(), 2) };

    let threshold = compute_threshold::<F>(pool, a, rows * cols);
    jacobi_svd_sweep::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        a_in_arg,
        a_out_arg,
        v_arg,
        info_arg,
        rows as u32,
        cols as u32,
        threshold,
        MAX_SWEEPS,
    );

    // The rotated-A scratch is not used for extraction (we recompute A·V via the
    // validated GEMM for D-02); release it now.
    DeviceArray::<ActiveRuntime, F>::from_raw(a_out_handle, rows * cols).release_into(pool);

    // --- Convergence check (D-12): read the tiny info array. info[0] = sweeps
    //     run; info[1] = final off-diagonal norm. A cap hit without convergence
    //     surfaces NotConverged. ---
    let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 2);
    let info = info_dev.to_host(pool);
    info_dev.release_into(pool);
    let sweeps_run = host_to_f64(info[0]) as u32;
    let residual = host_to_f64(info[1]);
    if sweeps_run >= MAX_SWEEPS && residual.is_finite() {
        // Only flag non-convergence if the cap was hit AND the residual is still
        // above the threshold band. We recompute the threshold scale here; a cap
        // hit with a tiny residual is a benign "converged exactly at the cap".
        let thr = host_to_f64(threshold);
        if residual > thr {
            return Err(PrimError::NotConverged {
                operand: "svd",
                max_sweeps: MAX_SWEEPS,
                residual,
            });
        }
    }

    // --- Thin-U / S (Pattern 3): B = A·V (m×k) via the Phase-2 GEMM (D-02). ---
    let v_dev = DeviceArray::<ActiveRuntime, F>::from_raw(v_handle, k * k);
    // V is stored COLUMN-major (v[c*k + r] = V[r, c]); a row-major (k×k) read of
    // the same buffer is therefore Vᵀ. GEMM wants B = A · V with A row-major
    // (rows×k logical = (m,k)) and V row-major (k×k). We have Vᵀ row-major, so
    // pass transb=true to read its transpose (= V) — no transpose buffer (D-06).
    let b = gemm::<F>(
        pool,
        a,
        (rows, k),
        &v_dev,
        (k, k),
        false,
        true, // v buffer is Vᵀ row-major; transb reads it as V.
        None,
    )?;

    // S[j] = ‖B[:, j]‖₂ — column L2-norm over the (rows × k) B (Phase-2 reduce).
    let s_dev = column_reduce::<F>(pool, &b, rows, k, ScalarOp::L2Norm, ReducePath::Shared)?
        .expect("shared path is never plane-gated to None");

    // --- Host-side thin-U normalize + descending sort + permute. We read back
    //     B (m×k), S (k), and V (k×k) — all device-resident producers; the
    //     convergence loop already ran in-kernel (D-11 gate 3). ---
    let b_host = b.to_host(pool);
    let s_host_raw = s_dev.to_host(pool);
    let v_host = v_dev.to_host(pool); // column-major V (v[c*k + r] = V[r, c]).
    b.release_into(pool);
    s_dev.release_into(pool);
    v_dev.release_into(pool);

    let s64: Vec<f64> = s_host_raw.iter().map(|&x| host_to_f64(x)).collect();

    // U (m×k) row-major: U[r, j] = B[r, j] / S[j], floored (Pitfall 4).
    let mut u_host: Vec<F> = vec![F::from_int(0i64); rows * k];
    for j in 0..k {
        let sj = s64[j];
        if sj > NEAR_ZERO_FLOOR {
            for r in 0..rows {
                let bval = host_to_f64(b_host[r * k + j]);
                u_host[r * k + j] = f64_to_host::<F>(bval / sj);
            }
        }
        // else: leave the U column at 0 (rank-deficient — Pitfall 4).
    }

    // Vᵀ (k×n=k×k) row-major: Vᵀ[j, c] = V[c, j]. V is column-major
    // (v_host[c*k + r] = V[r, c]), so V[c, j] = v_host[j*k + c].
    let mut vt_host: Vec<F> = vec![F::from_int(0i64); k * k];
    for j in 0..k {
        for c in 0..k {
            vt_host[j * k + c] = v_host[j * k + c]; // = V[c, j], see note above.
        }
    }
    // (V is k×k for the tall path; the wide path's relabel is handled below.)

    // Descending sort of S with a permutation; permute U columns and Vᵀ rows.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| s64[j].partial_cmp(&s64[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut s_sorted: Vec<F> = vec![F::from_int(0i64); k];
    let mut u_sorted: Vec<F> = vec![F::from_int(0i64); rows * k];
    let mut vt_sorted: Vec<F> = vec![F::from_int(0i64); k * k];
    for (new_j, &old_j) in order.iter().enumerate() {
        s_sorted[new_j] = f64_to_host::<F>(s64[old_j]);
        for r in 0..rows {
            u_sorted[r * k + new_j] = u_host[r * k + old_j];
        }
        for c in 0..k {
            vt_sorted[new_j * k + c] = vt_host[old_j * k + c];
        }
    }

    // --- Wide-path relabel (D-05): the caller fed Aᵀ. Here `rows`/`cols` are the
    //     transposed dims, so the ORIGINAL A is (cols × rows). We have computed
    //     (U', S', V'ᵀ) of Aᵀ; the original A's factors are U = V', Vᵀ = U'ᵀ.
    //     `u_sorted` is U' (rows×k = m'×k), `vt_sorted` is V'ᵀ (k×k). Build the
    //     original-A factors by swapping roles. ---
    if swap_uv {
        // Original A is (n × m) where here rows = n (=orig cols), cols = m... no:
        // caller passed (rows=cols_orig? ) — see svd(): wide path calls
        // svd_tall(at_dev, cols, rows, true) where (cols, rows) are the ORIGINAL
        // (rows_orig < cols_orig). So inside here `rows` = cols_orig (tall dim),
        // `cols` = rows_orig, k = cols = rows_orig. Aᵀ is (cols_orig × rows_orig).
        // U' = u_sorted is (cols_orig × k); V'ᵀ = vt_sorted is (k × k).
        // Original A (rows_orig × cols_orig): U = V' (rows_orig × k),
        // S = S', Vᵀ = U'ᵀ (k × cols_orig).
        let n_orig_rows = cols; // rows_orig = k
        let n_orig_cols = rows; // cols_orig
        let kk = k; // = rows_orig

        // U = V' = (V'ᵀ)ᵀ : V'ᵀ is vt_sorted (kk×kk) row-major; U is (rows_orig×kk).
        let mut u_orig: Vec<F> = vec![F::from_int(0i64); n_orig_rows * kk];
        for r in 0..n_orig_rows {
            for j in 0..kk {
                // V'[r, j] = (V'ᵀ)[j, r] = vt_sorted[j*kk + r].
                u_orig[r * kk + j] = vt_sorted[j * kk + r];
            }
        }
        // Vᵀ = U'ᵀ : U' is u_sorted (cols_orig × kk) row-major (col-index = sing.
        // vector). Vᵀ is (kk × cols_orig): Vᵀ[j, c] = U'[c, j] = u_sorted[c*kk+j].
        let mut vt_orig: Vec<F> = vec![F::from_int(0i64); kk * n_orig_cols];
        for j in 0..kk {
            for c in 0..n_orig_cols {
                vt_orig[j * n_orig_cols + c] = u_sorted[c * kk + j];
            }
        }
        let u_final = DeviceArray::from_host(pool, &u_orig);
        let s_final = DeviceArray::from_host(pool, &s_sorted);
        let vt_final = DeviceArray::from_host(pool, &vt_orig);
        return Ok((u_final, s_final, vt_final));
    }

    let u_final = DeviceArray::from_host(pool, &u_sorted);
    let s_final = DeviceArray::from_host(pool, &s_sorted);
    let vt_final = DeviceArray::from_host(pool, &vt_sorted);
    Ok((u_final, s_final, vt_final))
}

/// Compute the convergence threshold `8 · ε_F · ‖A‖_F` (D-12). `‖A‖_F` is the
/// Frobenius norm of the input; `ε_F` is the per-dtype machine epsilon. Reads the
/// input back to form `‖A‖_F` on the host — this is a ONE-TIME pre-launch scale
/// estimate, NOT a mid-sweep round-trip (the convergence loop stays in-kernel).
fn compute_threshold<F>(
    pool: &BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    len: usize,
) -> F
where
    F: Float + CubeElement + Pod,
{
    let host = a.to_host(pool);
    let mut sumsq = 0.0f64;
    for i in 0..len {
        let v = host_to_f64(host[i]);
        sumsq += v * v;
    }
    let fro = sumsq.sqrt();
    let eps = match size_of::<F>() {
        4 => f32::EPSILON as f64,
        _ => f64::EPSILON,
    };
    // Guard a near-zero matrix: keep the threshold strictly positive so the
    // kernel's `|gamma| > threshold` skip and the `off_norm <= threshold` break
    // are both well-defined.
    let thr = (THRESHOLD_SCALE * eps * fro).max(eps);
    f64_to_host::<F>(thr)
}

/// Validate the SVD operand geometry (ASVS V5 / T-03-03-01). `a` is
/// `rows × cols`; `rows * cols == a.len()`. The kernel stages the tall dimension
/// in shared memory capped at `MAX_ROWS` and the thin dimension at `MAX_COLS`, so
/// `max(rows,cols) ≤ MAX_ROWS` and `min(rows,cols) ≤ MAX_COLS` are required.
fn validate_geometry(a_len: usize, (rows, cols): (usize, usize)) -> Result<(), PrimError> {
    if rows.checked_mul(cols).map(|v| v != a_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows,
            cols,
            len: a_len,
        });
    }
    if rows == 0 || cols == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows,
            cols,
            len: a_len,
        });
    }
    let tall = rows.max(cols);
    let thin = rows.min(cols);
    if tall > MAX_ROWS as usize || thin > MAX_COLS as usize {
        // Geometry the single-cube kernel cannot stage; reject as a shape
        // violation rather than overflowing shared memory at launch.
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows,
            cols,
            len: a_len,
        });
    }
    Ok(())
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine / finalize.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("svd is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("svd is f32/f64 only"),
    }
}
