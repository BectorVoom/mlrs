//! Symmetric eigendecomposition host API (PRIM-05) — `A = V·diag(w)·Vᵀ` for a
//! square SYMMETRIC `A` (`n×n`), returning the device-resident eigenvalues `w`
//! (length `n`, DESCENDING — D-04) and eigenvectors `V` (`n×n`, eigenvectors as
//! columns). Drives the two-sided cyclic Jacobi sweep kernel
//! ([`mlrs_kernels::jacobi_eig_sweep`]) for the diagonalisation, then sorts the
//! converged diagonal descending on the host.
//!
//! ## The covariance/Gram feeder + buffer reuse (D-11 gate 2)
//! The eig path's only v1 feeder is the symmetric-by-construction covariance
//! Gram (`prims/covariance.rs`), so `A` is TRUSTED symmetric (D-06): this API
//! validates SQUARENESS but never forms `(A+Aᵀ)/2`. The optional `out` buffer
//! lets the caller thread the covariance/GEMM output handle straight through as
//! the kernel's working input — mirroring covariance's own "Gram reuses the GEMM
//! buffer" reuse — so the `full` PCA path does not allocate a parallel `n²`
//! matrix (D-11 gate 2, load-bearing for the Plan-05 memory gate). When `out` is
//! supplied it is copied into the kernel's `a_in` working buffer (the kernel
//! writes only `w`/`V`, leaving the caller's buffer the eig INPUT); when `None`
//! the input array is used directly.
//!
//! ## In-kernel convergence (D-11 gate 3)
//! The two-sided sweep loop — including the off-diagonal-norm convergence test —
//! runs entirely inside the single kernel launch (NO host round-trip between
//! sweeps). This API reads back ONLY the tiny length-`n` eigenvalue diagonal,
//! the `n×n` `V`, and the length-2 info array for the host-side descending sort
//! + the convergence check; it performs no read-back of intermediate sweeps.
//!
//! ## Descending sort (D-04) + convergence failure (D-12)
//! `np.linalg.eigh` returns eigenvalues ASCENDING; the device eig sorts them
//! DESCENDING so estimators inherit the right order. The host performs an `O(n)`
//! selection sort of the converged diagonal post-convergence (A4 — this is the
//! final sort, NOT the convergence loop the D-11 gate 3 concerns) and permutes
//! the eigenvector columns to match. If the kernel hit the sweep cap without
//! driving the off-diagonal norm below threshold, this API returns
//! [`PrimError::NotConverged`] rather than a silently-unconverged result.
//!
//! Tests live in `crates/mlrs-backend/tests/eig_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64};
use mlrs_core::PrimError;
use mlrs_kernels::jacobi_eig_sweep;
use mlrs_kernels::MAX_DIM;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Off-diagonal threshold scale factor `c` in `conv_thr = c · ε_F · ‖A‖_F ·
/// sqrt(pairs)` (D-12). `8` holds 1e-5 across the D-08 sweep while staying
/// reachable in f32 (mirrors the SVD host).
const THRESHOLD_SCALE: f64 = 8.0;

/// Max-sweep cap (D-12). Cyclic Jacobi converges quadratically (~10 sweeps for
/// the small symmetric covariance Gram); 30 is generous headroom (Pitfall 5).
const MAX_SWEEPS: u32 = 30;

/// Compute the symmetric eigendecomposition of `a` (`n × n`, row-major,
/// TRUSTED symmetric — D-06), returning the device-resident `(w, V)`: `w` the
/// length-`n` eigenvalues DESCENDING (D-04), `V` the `n × n` eigenvector matrix
/// (column-major, eigenvectors as columns).
///
/// - `a` is the row-major `n × n` symmetric matrix. Squareness is validated
///   (`a.len() == n*n`, and `n ≤ MAX_DIM`) BEFORE any unsafe launch (ASVS V5 /
///   T-03-04-01); a non-square geometry returns [`PrimError::NotSquare`].
/// - `out`, when supplied, is the covariance/GEMM output buffer reused as the
///   kernel's working input (D-11 gate 2): it must be the `n × n` operand. When
///   `None`, `a` is used directly.
/// - Non-convergence within the sweep cap returns [`PrimError::NotConverged`].
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log` (f64 runs on cpu,
/// skips on rocm — D-07).
#[allow(clippy::type_complexity)]
pub fn eig<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-03-04-01: validate squareness BEFORE any unsafe launch
    //     (D-06 trusts symmetry but validates the shape). ---
    validate_geometry(a.len(), n, out.as_ref().map(DeviceArray::len))?;

    let elem = size_of::<F>();

    // --- Working input buffer (D-11 gate 2 — covariance/GEMM buffer reuse).
    //     When the caller threads `out` through (the covariance Gram handle),
    //     that buffer is the kernel's `a_in` working input — no parallel n²
    //     allocation. The kernel only READS a_in (it writes w/V), so the
    //     caller's buffer is left intact as the eig input. When `out` is None
    //     we read directly from `a`. ---
    let (a_in_handle, a_in_owned) = match out {
        Some(buf) => (buf.handle().clone(), Some(buf)),
        None => (a.handle().clone(), None),
    };

    // --- Acquire the device-resident outputs: w (length n), V (n×n col-major),
    //     and the tiny info array [sweeps, residual]. ---
    let w_handle = pool.acquire(n * elem);
    let v_handle = pool.acquire(n * n * elem);
    let info_handle = pool.acquire(2 * elem);

    let client = pool.client().clone();
    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim {
        x: n as u32,
        y: 1,
        z: 1,
    };

    let (skip_thr, conv_thr) = compute_thresholds::<F>(pool, a, n * n, n);

    // --- cpu arm (EIG-PERF-CPU): run the SAME sweep on the host. -------------
    // The kernel launches ONE cube of `n` units with a `sync_cube` inside the
    // per-round loop. On `cubecl-cpu` that is `n` OS THREADS (60 of them at
    // `n = 60`) spin-waiting at ~`n · MAX_SWEEPS` barriers on a machine with far
    // fewer cores, at LLVM `-O0` — the same GPU-shaped-kernel pathology the KNN
    // and HDBSCAN cpu passes hit. It made `Umap::fit` at `n <= 64` (the spectral
    // -init path) take minutes. [`host_jacobi_eig`] replays the identical
    // schedule serially in native code; see it for why the result is unchanged.
    if host_eig_applicable() {
        // The working-input handle is either the caller's `out` buffer (released
        // just below via `a_in_owned`, exactly as the device arm does) or a
        // ref-counted clone of `a`'s. Wrapping it to read it back does not
        // transfer ownership — `DeviceArray` frees nothing on drop; the buffer
        // returns to the pool only through an explicit `release_into`.
        let a_in = DeviceArray::<ActiveRuntime, F>::from_raw(a_in_handle, n * n);
        let a_host: Vec<f64> = a_in.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        drop(a_in);
        if let Some(buf) = a_in_owned {
            buf.release_into(pool);
        }
        pool.release(w_handle, n * elem);
        pool.release(v_handle, n * n * elem);
        pool.release(info_handle, 2 * elem);

        let (w64, v64, sweeps_run, residual) = host_jacobi_eig(
            &a_host,
            n,
            host_to_f64(skip_thr),
            host_to_f64(conv_thr),
            MAX_SWEEPS,
        );
        if sweeps_run >= MAX_SWEEPS && residual.is_finite() && residual > host_to_f64(conv_thr) {
            return Err(PrimError::NotConverged {
                operand: "eig",
                max_sweeps: MAX_SWEEPS,
                residual,
            });
        }
        let (w_sorted, v_sorted) = sort_descending::<F>(&w64, &v64, n);
        return Ok((
            DeviceArray::from_host(pool, &w_sorted),
            DeviceArray::from_host(pool, &v_sorted),
        ));
    }

    // SAFETY: lengths are the carried/validated element counts (n*n, n, n*n, 2),
    // NEVER raw caller geometry; the kernel bounds every loop by the runtime `n`
    // and idles units with `i >= n` (mitigates T-03-04-01 / T-03-04-03, the OOB
    // device-read threat, ASVS V5).
    let a_in_arg = unsafe { ArrayArg::from_raw_parts(a_in_handle, n * n) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_handle.clone(), n) };
    let v_arg = unsafe { ArrayArg::from_raw_parts(v_handle.clone(), n * n) };
    let info_arg = unsafe { ArrayArg::from_raw_parts(info_handle.clone(), 2) };

    jacobi_eig_sweep::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        a_in_arg,
        w_arg,
        v_arg,
        info_arg,
        n as u32,
        skip_thr,
        conv_thr,
        MAX_SWEEPS,
    );

    // The reused `out` working buffer (if any) is now consumed by the launch;
    // release it back to the pool (the kernel only read it). When `out` was
    // None we never owned `a`, so nothing to release here.
    if let Some(buf) = a_in_owned {
        buf.release_into(pool);
    }

    // --- Convergence check (D-12): read the tiny info array. info[0] = sweeps
    //     run; info[1] = final off-diagonal norm. A cap hit without convergence
    //     surfaces NotConverged. ---
    let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 2);
    let info = info_dev.to_host(pool);
    info_dev.release_into(pool);
    let sweeps_run = host_to_f64(info[0]) as u32;
    let residual = host_to_f64(info[1]);
    if sweeps_run >= MAX_SWEEPS && residual.is_finite() && residual > host_to_f64(conv_thr) {
        // Release the converged-output handles before surfacing the error.
        DeviceArray::<ActiveRuntime, F>::from_raw(w_handle, n).release_into(pool);
        DeviceArray::<ActiveRuntime, F>::from_raw(v_handle, n * n).release_into(pool);
        return Err(PrimError::NotConverged {
            operand: "eig",
            max_sweeps: MAX_SWEEPS,
            residual,
        });
    }

    // --- Host-side descending sort (D-04) + eigenvector-column permute. We read
    //     back the small w (length n) and V (n×n) — both device-resident
    //     producers; the convergence loop already ran in-kernel (D-11 gate 3).
    //     This O(n) sort is the FINAL ordering, not the convergence loop. ---
    let w_dev = DeviceArray::<ActiveRuntime, F>::from_raw(w_handle, n);
    let v_dev = DeviceArray::<ActiveRuntime, F>::from_raw(v_handle, n * n);
    let w_host = w_dev.to_host(pool);
    let v_host = v_dev.to_host(pool); // column-major V (v[c*n + r] = V[r, c]).
    w_dev.release_into(pool);
    v_dev.release_into(pool);

    let w64: Vec<f64> = w_host.iter().map(|&x| host_to_f64(x)).collect();
    let v64: Vec<f64> = v_host.iter().map(|&x| host_to_f64(x)).collect();

    let (w_sorted, v_sorted) = sort_descending::<F>(&w64, &v64, n);
    let w_final = DeviceArray::from_host(pool, &w_sorted);
    let v_final = DeviceArray::from_host(pool, &v_sorted);
    Ok((w_final, v_final))
}

/// Order the converged spectrum DESCENDING (D-04) and permute `V`'s columns to
/// match, narrowing both back to `F`.
///
/// `w` is the unsorted diagonal; `v` is `V` in COLUMN-major layout
/// (`v[c*n + r] = V[r, c]`), which the permutation preserves. Shared by the
/// device and host arms so the two cannot drift in ordering or tie handling.
fn sort_descending<F>(w: &[f64], v: &[f64], n: usize) -> (Vec<F>, Vec<F>)
where
    F: Float + CubeElement + Pod,
{
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| w[j].partial_cmp(&w[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut w_sorted: Vec<F> = vec![F::from_int(0i64); n];
    let mut v_sorted: Vec<F> = vec![F::from_int(0i64); n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        w_sorted[new_j] = f64_to_host::<F>(w[old_j]);
        for r in 0..n {
            v_sorted[new_j * n + r] = f64_to_host::<F>(v[old_j * n + r]);
        }
    }
    (w_sorted, v_sorted)
}

/// Should the symmetric eigendecomposition run on the host?
///
/// True on `cpu` only. `MLRS_EIG_HOST=0` forces the device kernel back on for
/// on-target A/B; `=1` cannot force the host path onto a non-cpu backend, where
/// the single-cube kernel is a genuine parallel launch and this serial loop
/// would be a large regression.
fn host_eig_applicable() -> bool {
    crate::capability::active_backend_name() == "cpu"
        && crate::abflag::var("MLRS_EIG_HOST")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// Host twin of [`jacobi_eig_sweep`] — the SAME two-sided cyclic Jacobi, in
/// native scalar code (EIG-PERF-CPU).
///
/// Returns `(w, V, sweeps_run, final_off_diag_norm)` with `w` the UNSORTED
/// diagonal and `V` COLUMN-major, i.e. exactly what the kernel writes to
/// `w_out` / `v_out` / `info_out`, so the caller's sort + convergence check is
/// shared verbatim.
///
/// ## Why the result is the kernel's result
/// Every step of the kernel is replayed here in the same order: the same
/// even-padded circle-method pair schedule, the same `θ / t / c / s` rotation
/// from the same 2×2 block, the same two phases separated where the kernel puts
/// its barrier, the same per-row off-diagonal sums reduced by the same
/// halving tree, and the same `skip_thr` / `conv_thr` comparisons. Within a
/// round the kernel's units are genuinely order-independent — phase 1's pairs
/// own DISJOINT COLUMN pairs and phase 2's own disjoint ROW pairs, and neither
/// reads a location another pair writes in the same phase — so walking those
/// pairs sequentially computes the identical floating-point values rather than
/// merely an equivalent decomposition. (This is what makes the host arm a
/// drop-in and not a second algorithm with its own sign and ordering
/// conventions.)
fn host_jacobi_eig(
    a_in: &[f64],
    n: usize,
    skip_thr: f64,
    conv_thr: f64,
    max_sweeps: u32,
) -> (Vec<f64>, Vec<f64>, u32, f64) {
    // A row-major, V row-major (transposed to column-major at the end, as the
    // kernel's write-back does).
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Even-padded player count for the circle method (the kernel's `players`).
    let players = if n % 2 != 0 { n + 1 } else { n };
    let n_steps = players.saturating_sub(1);
    let half = players / 2;

    let mut off_norm = 0.0f64;
    let mut sweep = 0u32;
    let mut converged = false;
    // Per-pair rotations for the current round, carried from phase 1 to phase 2
    // exactly as each kernel unit carries `cs`/`sn` in registers across the
    // barrier (NEVER re-derived from `a_pq`, which phase 1 has already rotated).
    let mut round: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(half);

    while sweep < max_sweeps && !converged {
        for step in 0..n_steps {
            round.clear();

            // --- Phase 1 (A ← A·J, V ← V·J): column pairs, disjoint per pair.
            for pos in 0..half {
                let col_a = circle_player(pos, step, players);
                let col_b = circle_player(players - 1 - pos, step, players);
                let (lo, hi) = if col_a < col_b {
                    (col_a, col_b)
                } else {
                    (col_b, col_a)
                };
                if lo == hi || hi >= n {
                    continue; // self-pair, or a pairing touching the ghost player
                }

                let a_pp = a[lo * n + lo];
                let a_qq = a[hi * n + hi];
                let a_pq = a[lo * n + hi];
                if !(a_pq.abs() > skip_thr) {
                    continue;
                }

                let theta = (a_qq - a_pp) / (2.0 * a_pq);
                let denom = theta.abs() + (1.0 + theta * theta).sqrt();
                let mut t = 1.0 / denom;
                if theta < 0.0 {
                    t = -t;
                }
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = cs * t;

                for k in 0..n {
                    let a_kp = a[k * n + lo];
                    let a_kq = a[k * n + hi];
                    a[k * n + lo] = cs * a_kp - sn * a_kq;
                    a[k * n + hi] = sn * a_kp + cs * a_kq;
                }
                for r in 0..n {
                    let v_rp = v[r * n + lo];
                    let v_rq = v[r * n + hi];
                    v[r * n + lo] = cs * v_rp - sn * v_rq;
                    v[r * n + hi] = sn * v_rp + cs * v_rq;
                }
                round.push((lo, hi, cs, sn));
            }

            // --- Phase 2 (A ← Jᵀ·A): row pairs, disjoint per pair. This is the
            //     kernel's post-barrier half — it needs the FULLY phase-1-updated
            //     matrix, which is why it cannot be folded into the loop above.
            for &(lo, hi, cs, sn) in &round {
                for kk in 0..n {
                    let a_pk = a[lo * n + kk];
                    let a_qk = a[hi * n + kk];
                    a[lo * n + kk] = cs * a_pk - sn * a_qk;
                    a[hi * n + kk] = sn * a_pk + cs * a_qk;
                }
            }
        }

        // --- Convergence: per-row off-diagonal sums, then the kernel's halving
        //     tree reduction (replicated so the summation ORDER — and therefore
        //     the sweep at which f32 stops — matches).
        let mut off = vec![0.0f64; n.max(1)];
        for i in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                if j != i {
                    let aij = a[i * n + j];
                    acc += aij * aij;
                }
            }
            off[i] = acc;
        }
        let mut s = next_pow2_half(n);
        while s > 0 {
            for i in 0..s {
                if i + s < n {
                    off[i] += off[i + s];
                }
            }
            s /= 2;
        }
        off_norm = off[0].sqrt();
        if off_norm <= conv_thr {
            converged = true;
        }
        sweep += 1;
    }

    // Write-back: the diagonal as eigenvalues, V transposed into column-major.
    let mut w = vec![0.0f64; n];
    let mut v_col = vec![0.0f64; n * n];
    for i in 0..n {
        w[i] = a[i * n + i];
        for r in 0..n {
            v_col[i * n + r] = v[r * n + i];
        }
    }
    (w, v_col, sweep, off_norm)
}

/// Circle-method player index — the host twin of `jacobi_eig::circle_player`.
/// Position 0 is the fixed pivot; positions `1..players` rotate.
#[inline]
fn circle_player(pos: usize, step: usize, players: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let m = players - 1;
    ((pos - 1 + step) % m) + 1
}

/// Largest power of two strictly below `n` (0 for `n <= 1`) — the host twin of
/// `jacobi_eig::next_pow2_half`, so the reduction tree has the same shape.
#[inline]
fn next_pow2_half(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut s = 1usize;
    while s * 2 < n {
        s *= 2;
    }
    s
}

/// Validate the eig operand geometry (ASVS V5 / T-03-04-01). `a` must be a
/// square `n × n`: `a.len() == n*n`. The single-cube kernel stages the `n × n`
/// `A` + `V` in shared memory capped at `MAX_DIM`, so `n ≤ MAX_DIM` is required.
/// A non-square (or over-cap) geometry is rejected with [`PrimError::NotSquare`]
/// BEFORE any unsafe launch (D-06 trusts symmetry but validates squareness).
fn validate_geometry(a_len: usize, n: usize, out_len: Option<usize>) -> Result<(), PrimError> {
    // Squareness: a.len() must equal n*n. A mismatch means the caller's declared
    // order does not describe a square matrix.
    if n == 0 || n.checked_mul(n).map(|v| v != a_len).unwrap_or(true) {
        // Report the implied (rows, cols): a length that is not n*n is non-square.
        return Err(PrimError::NotSquare {
            operand: "eig",
            rows: n,
            cols: if n == 0 { 0 } else { a_len / n.max(1) },
        });
    }
    if n > MAX_DIM as usize {
        // Geometry the single-cube kernel cannot stage; reject rather than
        // overflow shared memory at launch.
        return Err(PrimError::NotSquare {
            operand: "eig",
            rows: n,
            cols: n,
        });
    }
    // The reused `out` buffer (D-11 gate 2) must itself be the n×n operand.
    if let Some(o) = out_len {
        if o != n * n {
            return Err(PrimError::NotSquare {
                operand: "eig.out",
                rows: n,
                cols: if n == 0 { 0 } else { o / n.max(1) },
            });
        }
    }
    Ok(())
}

/// Compute the `(skip_thr, conv_thr)` pair (D-12), mirroring the SVD host.
/// `‖A‖_F` is the input's Frobenius norm; `ε_F` the per-dtype machine epsilon;
/// `pairs = n(n-1)/2`.
///   - `skip_thr = ε_F · ‖A‖_F` — TINY, so rotations are essentially never
///     skipped (a loose skip bound stalls convergence — Pitfall 5).
///   - `conv_thr = 8 · ε_F · ‖A‖_F · sqrt(pairs)` — the convergence-break bound,
///     scaled by `sqrt(pairs)` to clear the ACCUMULATED f32 rounding floor.
/// Reads the input back ONCE to form `‖A‖_F` on the host — a pre-launch scale
/// estimate, NOT a mid-sweep round-trip (the convergence loop stays in-kernel).
fn compute_thresholds<F>(
    pool: &BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    len: usize,
    n: usize,
) -> (F, F)
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
    let pairs = (n * n.saturating_sub(1)) as f64 / 2.0;
    let skip_thr = (eps * fro).max(eps);
    let conv_thr = (THRESHOLD_SCALE * eps * fro * pairs.max(1.0).sqrt()).max(skip_thr);
    (f64_to_host::<F>(skip_thr), f64_to_host::<F>(conv_thr))
}
