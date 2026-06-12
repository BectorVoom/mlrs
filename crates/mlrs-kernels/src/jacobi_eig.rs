//! Two-sided cyclic Jacobi symmetric-eigendecomposition sweep kernel (PRIM-05)
//! — a single-cube, shared-memory `#[cube(launch)]` routine that diagonalises a
//! square SYMMETRIC matrix `A` (n×n) by a sequence of two-sided plane rotations
//! `A ← Jᵀ·A·J`, accumulating the rotations into `V`. On convergence the
//! off-diagonal entries of `A` are ~0: the diagonal carries the eigenvalues and
//! the accumulated columns of `V` are the eigenvectors, so the host
//! (`prims/eig.rs`) reads off `diag(A)` (UNSORTED — the host sorts descending,
//! D-04) and `V` without ever forming a second working matrix (D-11 gate 2).
//!
//! This is the two-sided sibling of [`crate::jacobi_svd`]: where the one-sided
//! SVD kernel applies each rotation to the COLUMNS of a tall `A` only
//! (`A ← A·J`), the symmetric eig kernel applies it on BOTH sides
//! (`A ← Jᵀ·A·J`), so a rotation of plane `(i, j)` touches rows `i, j` AND
//! columns `i, j` of `A`. The Jacobi angle is the classic symmetric formula on
//! the off-diagonal `a_ij`. Symmetry is TRUSTED (D-06): the kernel never forms
//! `(A + Aᵀ)/2`; the host validates squareness and feeds the symmetric-by-
//! construction covariance Gram.
//!
//! ## Single cube, in-kernel convergence loop (D-11 gate 3)
//! The whole sweep loop — including the off-diagonal-norm convergence test —
//! runs inside ONE kernel launch with NO host round-trip between sweeps (cubecl
//! 0.10 has cube-scoped `sync_cube()` but no portable device-wide barrier). The
//! eig matrices are SMALL (the covariance Gram is `n×n` with `n ≤ MAX_DIM`,
//! e.g. 4×4), so — unlike the SVD kernel where a `256×64` tile overflowed
//! gfx1100's 64 KiB LDS and `A` had to live in global — here BOTH `A` and `V`
//! fit comfortably in shared memory (`2 · MAX_DIM² · 4 B = 32 KiB` at the cap),
//! keeping every rotation read/write LDS-resident.
//!
//! ## Layout (one unit per row/column index)
//! Unit `i` (`UNIT_POS_X`) owns index `i`. The `n×n` symmetric `A` and the
//! `n×n` accumulator `V` are staged ROW-MAJOR in shared memory
//! (`a_sh[r*MAX_DIM + c]`, `v_sh[r*MAX_DIM + c]`), `V` initialised to the
//! identity. Per round-robin step a disjoint set of index pairs `(p, q)` rotate
//! concurrently — each pair is handled by the lower-indexed unit, which reads
//! the 2×2 block, computes the symmetric Jacobi rotation, and applies it to
//! rows/cols `p, q` of `A` and to columns `p, q` of `V`; the `sync_cube()`
//! between steps makes the writes visible before the next.
//!
//! ## Round-robin (chess-tournament) pair schedule (RESEARCH Pattern 6)
//! One sweep visits all `n(n-1)/2` index pairs over `n-1` steps, `floor(n/2)`
//! disjoint pairs per step (the standard circle method: index 0 fixed, the rest
//! rotate). Disjoint pairs touch disjoint indices, so the rotations within a
//! step are write-conflict-free. `continue` is NOT supported in `#[cube]`, so a
//! below-threshold pair is if-wrapped (`if a_pq.abs() > skip_thr { ... }`).
//!
//! ## Convergence (D-12 constants — RECORDED here)
//! TWO distinct thresholds are passed by value (the key Pitfall-5 fix, mirroring
//! the SVD kernel):
//!   - `skip_thr` — the per-pair rotation-skip bound. A pair `(p, q)` rotates
//!     only when `|a_pq| > skip_thr`. This stays TINY (`ε_F · ‖A‖_F`) so
//!     rotations are essentially never skipped; conflating it with the break
//!     bound stalls convergence.
//!   - `conv_thr` — the convergence-break bound. After each full sweep the
//!     off-diagonal Frobenius norm `sqrt(Σ_{i<j} a_ij²)` is reduced with a
//!     log₂-tree (in-kernel — mirrors `reduce_sumsq_shared`); the loop breaks
//!     when that norm `≤ conv_thr`. It is set to `8 · ε_F · ‖A‖_F · sqrt(pairs)`
//!     (`pairs = n(n-1)/2`) to clear the ACCUMULATED f32 rounding floor — a
//!     single-`ε` bound is unreachable in f32 for moderate `n` (Pitfall 5).
//!     `ε_f32 ≈ 1.2e-7`, `ε_f64 ≈ 2.2e-16`.
//! The loop also stops at `max_sweeps = 30` (generous; cyclic Jacobi converges
//! quadratically). A cap hit without convergence surfaces `NotConverged` on the
//! host (D-12).
//!
//! ## CubeCL expression notes
//! - `SharedMemory::<F>::new(N)` requires a COMPILE-TIME size — `a_sh` / `v_sh`
//!   are sized to the comptime cap (`MAX_DIM` × `MAX_DIM`) and the active region
//!   is bounded by the runtime `n` (mirrors `reduce.rs` sizing + `len` guard).
//! - `continue` is NOT supported in `#[cube]` — the "skip below-threshold pair"
//!   is `if a_pq.abs() > skip_thr { ...rotate... }` (RESEARCH Pattern 6).
//! - generic constants via `F::from_int` / `F::new`; `Float` methods `.abs()` /
//!   `.sqrt()`.
//! - No hardcoded plane width / 32 — the off-diagonal norm uses the shared-
//!   memory tree (not a plane path), avoiding plane-width portability concerns.
//!
//! Generic over `<F: Float + CubeElement>` and carries NO backend feature (D-13).
//! Per AGENTS.md §2 this file has NO in-source `mod tests` — the live launch
//! tests are in `crates/mlrs-backend/tests/eig_test.rs`.

use cubecl::prelude::*;

/// Comptime dimension cap for the shared `A` and `V` tiles (`MAX_DIM × MAX_DIM`)
/// and the off-diagonal accumulator. The eig order `n ≤ MAX_DIM`. At f32 this is
/// `64·64·4 = 16 KiB` for `A` + `16 KiB` for `V` + `64·4 = 256 B` for the
/// accumulator = ~32 KiB — within gfx1100's 64 KiB LDS. The host
/// (`prims/eig.rs`) rejects `n > MAX_DIM` before launch.
pub const MAX_DIM: u32 = 64;

/// Two-sided cyclic Jacobi sweep over a square symmetric `A` (`n × n`), staged
/// row-major in shared memory. Drives the off-diagonal entries to ~0 by
/// `A ← Jᵀ·A·J` and accumulates `J` into `V`. Writes the converged diagonal
/// (eigenvalues, UNSORTED — host sorts descending, D-04) to `w_out` and the
/// eigenvector matrix `V` (column-major) to `v_out`. `info_out[0]` carries the
/// sweep count actually run (so the host can detect a non-converged cap hit,
/// D-12 / `NotConverged`); `info_out[1]` carries the final off-diagonal norm.
///
/// - `a_in` is the row-major `n × n` symmetric input (`a_in[r*n + c]`); symmetry
///   is TRUSTED (no `(A+Aᵀ)/2` — D-06).
/// - `w_out` is the length-`n` diagonal (eigenvalues, unsorted).
/// - `v_out` is `V` in COLUMN-major layout (`v_out[c*n + r]`).
/// - `info_out` is length 2: `[sweeps_run, final_off_diag_norm]`.
/// - `n` is the runtime active dimension (`n ≤ MAX_DIM`).
/// - `skip_thr` is the TINY per-pair rotation-skip bound (`ε_F·‖A‖_F`).
/// - `conv_thr` is the off-diagonal-norm convergence-break bound
///   (`8·ε_F·‖A‖_F·sqrt(pairs)`).
/// - `max_sweeps` is the sweep cap (D-12).
///
/// Launch with ONE cube of `n` units (`CubeDim { x: n, .. }`).
#[cube(launch)]
pub fn jacobi_eig_sweep<F: Float + CubeElement>(
    a_in: &Array<F>,
    w_out: &mut Array<F>,
    v_out: &mut Array<F>,
    info_out: &mut Array<F>,
    n: u32,
    skip_thr: F,
    conv_thr: F,
    max_sweeps: u32,
) {
    // A and V staged row-major in shared (a_sh[r*MAX_DIM + c]); the matrices are
    // small so both fit LDS (unlike the SVD kernel, which kept A in global).
    let mut a_sh = SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize);
    let mut v_sh = SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize);
    // Per-pair off-diagonal contributions a_ij² for the convergence reduction.
    // One slot per index (the lower-indexed unit of a pair writes its a²).
    let mut off_sh = SharedMemory::<F>::new(MAX_DIM as usize);

    let i = UNIT_POS_X;
    let zero = F::from_int(0i64);
    let one = F::from_int(1i64);
    let two = F::from_int(2i64);

    // --- Stage: copy my row i of A (row-major in → row-major shared) and
    //     initialise my row i of V to the identity. Only the active region
    //     (i < n) participates; OOB units idle. ---
    if i < n {
        let mut c = 0u32;
        while c < n {
            a_sh[(i * MAX_DIM + c) as usize] = a_in[(i * n + c) as usize];
            let v = if c == i { one } else { zero };
            v_sh[(i * MAX_DIM + c) as usize] = v;
            c += 1u32;
        }
    }
    sync_cube();

    // --- Sweep loop (in-kernel; no host round-trip — D-11 gate 3). ---
    let mut sweep = 0u32;
    let mut converged = false;
    while sweep < max_sweeps && !converged {
        // Reset this sweep's per-pair off-diagonal accumulator.
        if i < n {
            off_sh[i as usize] = zero;
        }
        sync_cube();

        // Round-robin schedule: n-1 steps, each a disjoint set of pairs (circle
        // method — index 0 fixed, the rest rotate).
        let mut n_steps = 0u32;
        if n > 0u32 {
            n_steps = n - 1u32;
        }
        let mut step = 0u32;
        while step < n_steps {
            if i < n {
                let half = n / 2u32;
                let mut pos = 0u32;
                while pos < half {
                    // position -> player index under the circle rotation.
                    let idx_a = circle_player(pos, step, n);
                    let idx_b = circle_player(n - 1u32 - pos, step, n);
                    // Order the pair so (p, q) = (lo, hi).
                    let p = if idx_a < idx_b { idx_a } else { idx_b };
                    let q = if idx_a < idx_b { idx_b } else { idx_a };
                    // Only the LOW-index unit performs this pair's rotation.
                    if i == p && p != q {
                        // 2×2 symmetric block: a_pp, a_qq, a_pq (= a_qp).
                        let a_pp = a_sh[(p * MAX_DIM + p) as usize];
                        let a_qq = a_sh[(q * MAX_DIM + q) as usize];
                        let a_pq = a_sh[(p * MAX_DIM + q) as usize];

                        // Record this pair's off-diagonal contribution a_pq² for
                        // the convergence test (accumulate into the p slot).
                        off_sh[p as usize] += a_pq * a_pq;

                        // --- Symmetric Jacobi rotation that zeroes a_pq (skip
                        //     only when |a_pq| is below the TINY skip bound;
                        //     `continue` unsupported → if-wrap). ---
                        if a_pq.abs() > skip_thr {
                            // θ = (a_qq − a_pp) / (2·a_pq);
                            // t = sign(θ) / (|θ| + sqrt(1 + θ²));
                            // c = 1/sqrt(1 + t²);  s = c·t.
                            let theta = (a_qq - a_pp) / (two * a_pq);
                            let denom = theta.abs() + (one + theta * theta).sqrt();
                            let mut t = one / denom;
                            if theta < zero {
                                t = -t;
                            }
                            let cs = one / (one + t * t).sqrt();
                            let sn = cs * t;

                            // Apply Jᵀ·A·J. Update rows/cols p, q of A. First the
                            // two affected COLUMNS (k, p) and (k, q) for every row
                            // k, then the two affected ROWS (p, k) and (q, k).
                            // Because A is symmetric and we keep it symmetric, we
                            // update the full matrix. Read both column entries,
                            // write the rotated pair.
                            let mut k = 0u32;
                            while k < n {
                                let a_kp = a_sh[(k * MAX_DIM + p) as usize];
                                let a_kq = a_sh[(k * MAX_DIM + q) as usize];
                                a_sh[(k * MAX_DIM + p) as usize] = cs * a_kp - sn * a_kq;
                                a_sh[(k * MAX_DIM + q) as usize] = sn * a_kp + cs * a_kq;
                                k += 1u32;
                            }
                            // Now the affected ROWS (the column update above moved
                            // entries; apply the same rotation on the left).
                            let mut kk = 0u32;
                            while kk < n {
                                let a_pk = a_sh[(p * MAX_DIM + kk) as usize];
                                let a_qk = a_sh[(q * MAX_DIM + kk) as usize];
                                a_sh[(p * MAX_DIM + kk) as usize] = cs * a_pk - sn * a_qk;
                                a_sh[(q * MAX_DIM + kk) as usize] = sn * a_pk + cs * a_qk;
                                kk += 1u32;
                            }

                            // Accumulate the rotation into V columns p, q:
                            // V ← V·J (eigenvectors are the columns of V).
                            let mut r = 0u32;
                            while r < n {
                                let v_rp = v_sh[(r * MAX_DIM + p) as usize];
                                let v_rq = v_sh[(r * MAX_DIM + q) as usize];
                                v_sh[(r * MAX_DIM + p) as usize] = cs * v_rp - sn * v_rq;
                                v_sh[(r * MAX_DIM + q) as usize] = sn * v_rp + cs * v_rq;
                                r += 1u32;
                            }
                        }
                    }
                    pos += 1u32;
                }
            }
            sync_cube();
            step += 1u32;
        }

        // --- In-kernel off-diagonal-norm convergence test (no host round-trip).
        //     Tree-reduce off_sh[0..n] (= Σ_{i<j} a_ij²) into off_sh[0]. ---
        let mut s = next_pow2_half(n);
        while s > 0u32 {
            if i < s && (i + s) < n {
                let val = off_sh[(i + s) as usize];
                off_sh[i as usize] += val;
            }
            sync_cube();
            s /= 2u32;
        }
        let off_norm = off_sh[0usize].sqrt();
        if off_norm <= conv_thr {
            converged = true;
        }
        sync_cube();

        sweep += 1u32;
    }

    // --- Write back: the diagonal of A (eigenvalues, unsorted) to w_out, V
    //     (column-major) to v_out, plus the sweep count + final norm. ---
    if i < n {
        w_out[i as usize] = a_sh[(i * MAX_DIM + i) as usize];
        let mut r = 0u32;
        while r < n {
            // v_out column-major: column i, row r at i*n + r. v_sh is row-major
            // (v_sh[r*MAX_DIM + i] = V[r, i]).
            v_out[(i * n + r) as usize] = v_sh[(r * MAX_DIM + i) as usize];
            r += 1u32;
        }
    }
    if i == 0u32 {
        info_out[0usize] = F::cast_from(sweep);
        info_out[1usize] = off_sh[0usize].sqrt();
    }
}

/// Circle-method player index for circle position `pos` at rotation `step`, over
/// `n` players. Position 0 is the fixed pivot (player 0); positions `1..n`
/// rotate: player = ((pos - 1 + step) mod (n - 1)) + 1. This yields the standard
/// round-robin tournament so one sweep covers all `n(n-1)/2` pairs over `n-1`
/// steps. (Identical to the SVD kernel's helper — kept local so the two kernels
/// stay independent.)
#[cube]
fn circle_player(pos: u32, step: u32, n: u32) -> u32 {
    let mut player = 0u32;
    if pos > 0u32 {
        let m = n - 1u32;
        player = ((pos - 1u32 + step) % m) + 1u32;
    }
    player
}

/// Largest power of two strictly less than `n` (the starting stride for a
/// shared-memory tree reduction over `n` slots). For `n <= 1` returns 0 (no
/// reduction needed).
#[cube]
fn next_pow2_half(n: u32) -> u32 {
    let mut s = 1u32;
    while s * 2u32 < n {
        s *= 2u32;
    }
    let mut out = s;
    if n <= 1u32 {
        out = 0u32;
    }
    out
}
