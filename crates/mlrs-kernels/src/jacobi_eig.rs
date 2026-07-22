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
//! identity.
//!
//! ## Round-robin (chess-tournament) pair schedule, TWO-PHASE per round (LINEAR-01 perf lever)
//! Earlier revisions of this kernel visited the `n(n-1)/2` upper-triangle pairs
//! ONE AT A TIME with a SINGLE acting unit (`if i == 0u32 { ...whole rotation... }`)
//! doing the entire `O(n)` row+column+`V` update while the other `n-1` units
//! idled — `O(n²)` pairs fully serialized, i.e. `O(n³)` work per sweep on ONE
//! thread. Profiling `LinearRegression`'s Gram+eig path (`d=64`) showed `eig`
//! alone costing ~0.13-0.14s regardless of `n_samples` (a pure function of `d`),
//! ~17x more than at `d=16` — matching `C(64,2)/C(16,2) ≈ 16.8x`, i.e. the pair
//! COUNT, not any per-pair cost, confirming the serial schedule as the
//! bottleneck (`[[mlrs-linear-regression-optimization]]` memory).
//!
//! This revision reuses [`crate::jacobi_svd`]'s ALREADY-PROVEN ghost-padded
//! round-robin (circle-method) tournament schedule ([`circle_player`], CR-01):
//! each ROUND pairs up ALL `n` (real, padded to `players` with a ghost bye when
//! `n` is odd) indices into `players/2` MUTUALLY INDEX-DISJOINT pairs, so
//! `players - 1` rounds cover every one of the `n(n-1)/2` pairs exactly once.
//! The one-sided SVD kernel can rotate a round's disjoint pairs fully
//! concurrently because `A ← A·J` only ever touches COLUMNS, and disjoint
//! COLUMN pairs have disjoint write footprints. The two-sided `A ← Jᵀ·A·J`
//! does NOT have that property directly: rotating pair `(lo, hi)` writes rows
//! `lo, hi` (ALL columns) AND columns `lo, hi` (ALL rows) — including entries
//! like `A[lo, a]` for another pair `(a, b)`'s column `a`, so a naive
//! concurrent two-sided update of a whole round WOULD race.
//!
//! The fix is to split each round into two barrier-separated phases,
//! mirroring how `A ← Jᵀ·A·J` is mathematically `Jᵀ·(A·J)` — right-multiply,
//! THEN left-multiply:
//!   - **Phase 1 (right-multiply, `A ← A·J` + `V ← V·J`)**: the acting
//!     (lower-indexed) unit of EACH pair in the round computes its rotation
//!     from the OLD 2×2 block and updates ONLY columns `lo, hi` (all rows) of
//!     `A` and `V` — structurally IDENTICAL to the SVD kernel's single-phase
//!     update. Different pairs in a round own DISJOINT columns, so this phase
//!     is race-free across the WHOLE round (no cross-pair write ever touches
//!     another pair's columns, and no cross-pair READ depends on another
//!     pair's not-yet-updated columns, since a pair's angle only depends on
//!     its own 2×2 block).
//!   - **Barrier** (`sync_cube()`), then:
//!   - **Phase 2 (left-multiply, `A ← Jᵀ·A`)**: EACH pair's acting unit
//!     updates ROWS `lo, hi` (all columns) of `A`, reusing the SAME `cs`/`sn`
//!     computed in phase 1 (kept in per-unit registers across the barrier —
//!     NOT re-derived: re-reading `a[lo,hi]` here would see phase 1's
//!     already-near-zeroed value and wrongly skip the rotation). Different
//!     pairs own DISJOINT rows, so this phase is also race-free across the
//!     round; it correctly reads the FULLY phase-1-updated matrix (the
//!     intended intermediate `A·J_round`).
//!
//! This turns `O(n²)` serialized pairs into `O(n)` rounds of `O(n)` work each,
//! run by `players/2` CONCURRENT acting units per round instead of one — the
//! same complexity class as the already-parallel SVD kernel. `continue` is NOT
//! supported in `#[cube]`, so a below-threshold pair is if-wrapped
//! (`if a_pq.abs() > skip_thr { ... }`, RESEARCH Pattern 6) exactly as before.
//!
//! ## Convergence (D-12 constants — RECORDED here)
//! TWO distinct thresholds are passed by value (the key Pitfall-5 fix, mirroring
//! the SVD kernel):
//!   - `skip_thr` — the per-pair rotation-skip bound. A pair `(p, q)` rotates
//!     only when `|a_pq| > skip_thr`. This stays TINY (`ε_F · ‖A‖_F`) so
//!     rotations are essentially never skipped; conflating it with the break
//!     bound stalls convergence.
//!   - `conv_thr` — the convergence-break bound. After each full sweep the
//!     off-diagonal Frobenius norm is measured DIRECTLY from the current matrix
//!     state (each unit sums `a_ij²` over `j != i` for its row `i`, then a
//!     log₂-tree reduction — in-kernel, mirrors `reduce_sumsq_shared`). The
//!     per-row sums double-count each pair, so the reduced value is
//!     `2·Σ_{i<j} a_ij²` and its sqrt is `sqrt(2)·‖offdiag‖`; the loop breaks
//!     when that `≤ conv_thr`. `conv_thr` is `8 · ε_F · ‖A‖_F · sqrt(pairs)`
//!     (`pairs = n(n-1)/2`) to clear the ACCUMULATED f32 rounding floor — a
//!     single-`ε` bound is unreachable in f32 for moderate `n` (Pitfall 5); the
//!     extra `sqrt(2)` only makes the break marginally STRICTER (more accurate),
//!     which is safe. `ε_f32 ≈ 1.2e-7`, `ε_f64 ≈ 2.2e-16`.
//! The loop also stops at `max_sweeps = 30` (generous; cyclic Jacobi converges
//! quadratically — the round-robin reordering does not change WHICH pairs are
//! visited per sweep, only that they now run concurrently, so the sweep count
//! to convergence is unaffected). A cap hit without convergence surfaces
//! `NotConverged` on the host (D-12).
//!
//! ## CubeCL expression notes
//! - `SharedMemory::<F>::new(N)` requires a COMPILE-TIME size — `a_sh` / `v_sh`
//!   are sized to the comptime cap (`MAX_DIM` × `MAX_DIM`) and the active region
//!   is bounded by the runtime `n` (mirrors `reduce.rs` sizing + `len` guard).
//!   The round-robin schedule adds NO new shared memory (the per-pair `lo` /
//!   `hi` / `cs` / `sn` / flags are per-unit REGISTERS, not shared arrays) —
//!   the LDS budget is unchanged from the serial revision.
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
        // Ghost-padded round-robin (CR-01, the jacobi_svd_sweep precedent):
        // pad to an EVEN player count so the circle method enumerates ALL
        // n(n-1)/2 pairs for odd AND even `n`. `players = n` when n is even,
        // else `n + 1` (the ghost player >= n supplies the bye). Run
        // `players - 1` rounds of `players / 2` positions; SKIP any pairing
        // touching the ghost index (`hi >= n`).
        let mut players = n;
        if n % 2u32 != 0u32 {
            players = n + 1u32;
        }
        let mut n_steps = 0u32;
        if players > 0u32 {
            n_steps = players - 1u32;
        }

        let mut step = 0u32;
        while step < n_steps {
            // Per-unit pair identity for this round, found once and reused
            // across BOTH phases below (phase 2 must NOT re-derive `a_pq`
            // from the shared matrix — phase 1 already rotated it toward 0).
            let mut lo = 0u32;
            let mut hi = 0u32;
            let mut is_lo = false;
            let mut do_rot = false;
            let mut cs = one;
            let mut sn = zero;

            if i < n {
                let half = players / 2u32;
                let mut pos = 0u32;
                while pos < half {
                    // Circle-method pairing for this round: position pos <->
                    // position (players-1-pos), over the padded `players`.
                    let col_a = circle_player(pos, step, players);
                    let col_b = circle_player(players - 1u32 - pos, step, players);
                    let plo = if col_a < col_b { col_a } else { col_b };
                    let phi = if col_a < col_b { col_b } else { col_a };
                    // Only the LOW-index unit performs this pair's rotation,
                    // and only for REAL pairs (skip the self-pair and any
                    // pairing touching the ghost index `>= n`).
                    if i == plo && plo != phi && phi < n {
                        lo = plo;
                        hi = phi;
                        is_lo = true;
                    }
                    pos += 1u32;
                }

                // --- Phase 1 (right-multiply, A ← A·J + V ← V·J): compute
                //     the rotation from the OLD 2×2 block and update ONLY
                //     columns lo, hi (all rows) of A and V. Every pair in
                //     this round owns a DISJOINT column pair, so different
                //     pairs' phase-1 updates never write-alias (module docs)
                //     — this is what makes the whole round parallel, unlike
                //     the old fully-serial single-acting-unit schedule. ---
                if is_lo {
                    // 2×2 symmetric block: a_pp, a_qq, a_pq (= a_qp).
                    let a_pp = a_sh[(lo * MAX_DIM + lo) as usize];
                    let a_qq = a_sh[(hi * MAX_DIM + hi) as usize];
                    let a_pq = a_sh[(lo * MAX_DIM + hi) as usize];

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
                        cs = one / (one + t * t).sqrt();
                        sn = cs * t;
                        do_rot = true;

                        // A·J: column pass over all rows k, columns lo, hi.
                        let mut k = 0u32;
                        while k < n {
                            let a_kp = a_sh[(k * MAX_DIM + lo) as usize];
                            let a_kq = a_sh[(k * MAX_DIM + hi) as usize];
                            a_sh[(k * MAX_DIM + lo) as usize] = cs * a_kp - sn * a_kq;
                            a_sh[(k * MAX_DIM + hi) as usize] = sn * a_kp + cs * a_kq;
                            k += 1u32;
                        }

                        // V ← V·J: same column pass over V's rows.
                        let mut r = 0u32;
                        while r < n {
                            let v_rp = v_sh[(r * MAX_DIM + lo) as usize];
                            let v_rq = v_sh[(r * MAX_DIM + hi) as usize];
                            v_sh[(r * MAX_DIM + lo) as usize] = cs * v_rp - sn * v_rq;
                            v_sh[(r * MAX_DIM + hi) as usize] = sn * v_rp + cs * v_rq;
                            r += 1u32;
                        }
                    }
                }
            }
            // Barrier: phase 2 needs the FULLY phase-1-updated matrix (every
            // pair's column update in this round complete) before rotating
            // rows — see the module docs for why this can't be one phase.
            sync_cube();

            // --- Phase 2 (left-multiply, A ← Jᵀ·A): update ONLY rows lo, hi
            //     (all columns) of A, reusing the SAME cs/sn computed in
            //     phase 1 above (kept in this unit's registers across the
            //     barrier — NOT re-derived from a_pq, which phase 1 already
            //     rotated toward 0). Different pairs own DISJOINT rows, so
            //     this is race-free across the round exactly like phase 1. ---
            if is_lo && do_rot {
                let mut kk = 0u32;
                while kk < n {
                    let a_pk = a_sh[(lo * MAX_DIM + kk) as usize];
                    let a_qk = a_sh[(hi * MAX_DIM + kk) as usize];
                    a_sh[(lo * MAX_DIM + kk) as usize] = cs * a_pk - sn * a_qk;
                    a_sh[(hi * MAX_DIM + kk) as usize] = sn * a_pk + cs * a_qk;
                    kk += 1u32;
                }
            }
            sync_cube();
            step += 1u32;
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

/// Circle-method player index for circle position `pos` at round `step`, over
/// `players` positions (the EVEN-padded count — `n` or `n+1`, CR-01). Position
/// 0 is the fixed pivot; positions `1..players` rotate:
/// `player = ((pos - 1 + step) mod (players - 1)) + 1`. With an even `players`
/// the standard round-robin tournament covers all `players·(players-1)/2`
/// position pairs over `players - 1` rounds; the caller skips any pairing that
/// touches a ghost position (`>= n`), leaving exactly the `n·(n-1)/2` real
/// pairs. Byte-identical to `jacobi_svd_sweep`'s `circle_player` (kept as its
/// own copy per this crate's per-kernel-file convention — no cross-file
/// `#[cube]` fn imports).
#[cube]
fn circle_player(pos: u32, step: u32, players: u32) -> u32 {
    let mut player = 0u32;
    if pos > 0u32 {
        let m = players - 1u32;
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
