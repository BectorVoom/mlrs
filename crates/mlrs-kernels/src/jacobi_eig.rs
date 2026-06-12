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
//! ## Cyclic (sequential-pair) sweep schedule — NOT the SVD's parallel schedule
//! One sweep visits all `n(n-1)/2` upper-triangle index pairs `(p, q)` in cyclic
//! row-major order, ONE pair at a time. Unlike the one-sided SVD kernel (which
//! rotates index-disjoint COLUMN pairs concurrently because each rotation's
//! write footprint is its two columns), the TWO-sided eig rotation of plane
//! `(p, q)` touches the ENTIRE rows AND columns `p, q` — including cross entries
//! that belong to another index-disjoint pair — so index-disjoint pairs are NOT
//! footprint-disjoint and CANNOT rotate concurrently without racing. We instead
//! parallelise only the `O(n)` row/column/`V` update ACROSS units within each
//! pair (unit `k` updates index `k`'s entries), with a `sync_cube()` between the
//! column pass and the row pass and after each pair. `continue` is NOT supported
//! in `#[cube]`, so a below-threshold pair is if-wrapped
//! (`if a_pq.abs() > skip_thr { ... }`).
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
    //
    // ## Why pairs are processed SEQUENTIALLY (not the SVD's disjoint-parallel
    //    schedule)
    // The one-sided SVD kernel rotates disjoint COLUMN pairs concurrently: a
    // column-disjoint pair set has a disjoint write footprint (each rotation
    // touches only its two columns). The TWO-sided eig rotation of plane (p, q)
    // touches the ENTIRE rows AND columns p, q — including the cross entries
    // a[p][a], a[q][b] that belong to ANOTHER index-disjoint pair (a, b). So
    // index-disjoint pairs are NOT footprint-disjoint here, and rotating them
    // concurrently would race on those cross entries. We therefore visit the
    // n(n-1)/2 upper-triangle pairs (p, q) one at a time (cyclic row-major
    // order), parallelising only the O(n) row/column/V update ACROSS units
    // within each pair (unit k updates index k's entries). A sync_cube after the
    // angle read and after the writes keeps the single-pair update race-free.
    let mut sweep = 0u32;
    let mut converged = false;
    while sweep < max_sweeps && !converged {
        let mut p = 0u32;
        while p < n {
            let mut q = p + 1u32;
            while q < n {
                // A SINGLE unit (unit 0) performs the entire two-sided rotation
                // for plane (p, q) — reading/writing the full rows AND columns
                // p, q of A and the columns p, q of V — while all other units
                // idle. This mirrors the SVD kernel's "the acting unit does the
                // whole rotation" idiom (which is proven on the CPU backend) and
                // sidesteps the cross-unit shared-memory aliasing that a
                // distributed two-sided update would create (an index-disjoint
                // unit's column write would alias this plane's row footprint). A
                // sync_cube after the pair makes the result visible to all units
                // before the next pair / the convergence reduction.
                if i == 0u32 {
                    // 2×2 symmetric block: a_pp, a_qq, a_pq (= a_qp).
                    let a_pp = a_sh[(p * MAX_DIM + p) as usize];
                    let a_qq = a_sh[(q * MAX_DIM + q) as usize];
                    let a_pq = a_sh[(p * MAX_DIM + q) as usize];

                    // --- Symmetric Jacobi rotation that zeroes a_pq (skip only
                    //     when |a_pq| is below the TINY skip bound; `continue`
                    //     unsupported → if-wrap). ---
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

                        // Apply A ← Jᵀ·A·J. First A·J (column pass over all rows
                        // k: columns p, q), then Jᵀ·(A·J) (row pass over all
                        // columns k: rows p, q). The single acting unit does both
                        // passes in order, so no intra-pair barrier is needed.
                        let mut k = 0u32;
                        while k < n {
                            let a_kp = a_sh[(k * MAX_DIM + p) as usize];
                            let a_kq = a_sh[(k * MAX_DIM + q) as usize];
                            a_sh[(k * MAX_DIM + p) as usize] = cs * a_kp - sn * a_kq;
                            a_sh[(k * MAX_DIM + q) as usize] = sn * a_kp + cs * a_kq;
                            k += 1u32;
                        }
                        let mut kk = 0u32;
                        while kk < n {
                            let a_pk = a_sh[(p * MAX_DIM + kk) as usize];
                            let a_qk = a_sh[(q * MAX_DIM + kk) as usize];
                            a_sh[(p * MAX_DIM + kk) as usize] = cs * a_pk - sn * a_qk;
                            a_sh[(q * MAX_DIM + kk) as usize] = sn * a_pk + cs * a_qk;
                            kk += 1u32;
                        }

                        // Accumulate the rotation into V columns p, q (V ← V·J;
                        // eigenvectors are the columns of V).
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
                sync_cube();
                q += 1u32;
            }
            p += 1u32;
        }

        // --- In-kernel off-diagonal-norm convergence test (no host round-trip).
        //     Measure the TRUE post-sweep off-diagonal norm directly from the
        //     current matrix state: unit i sums a_ij² over j != i for its row i
        //     into off_sh[i] (a real measurement of where the matrix stands after
        //     this sweep's rotations — NOT an in-sweep estimate that a later
        //     rotation could refill). The per-row sums double-count each pair
        //     (a_ij² appears in row i and row j), so off_sh[0] holds
        //     2·Σ_{i<j} a_ij²; the scalar factor is folded consistently into the
        //     conv_thr comparison (the host scales conv_thr the same way). ---
        if i < n {
            let mut acc = zero;
            let mut j = 0u32;
            while j < n {
                if j != i {
                    let aij = a_sh[(i * MAX_DIM + j) as usize];
                    acc += aij * aij;
                }
                j += 1u32;
            }
            off_sh[i as usize] = acc;
        }
        sync_cube();

        // Tree-reduce off_sh[0..n] (= 2·Σ_{i<j} a_ij²) into off_sh[0].
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
