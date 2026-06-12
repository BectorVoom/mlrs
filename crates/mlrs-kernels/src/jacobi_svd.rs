//! One-sided (Hestenes) Jacobi SVD sweep kernel (PRIM-05) — a single-cube,
//! shared-memory `#[cube(launch)]` routine that orthogonalizes the columns of a
//! tall matrix `A` (m×n, m ≥ n) by a sequence of right-applied plane rotations
//! `A ← A·J`, accumulating the rotations into `V`. On convergence the rotated
//! columns of `A` are mutually orthogonal: column `j` has 2-norm `σ_j` and
//! direction `u_j`, so the host (`prims/svd.rs`) recovers the thin factors
//! `U = (A·V)/S`, `S`, `Vᵀ` without ever forming a square `U` (D-02).
//!
//! ## Single cube, in-kernel convergence loop (D-11 gate 3)
//! The whole sweep loop — including the off-diagonal-norm convergence test —
//! runs inside ONE kernel launch with NO host round-trip between sweeps. cubecl
//! 0.10 has cube-scoped `sync_cube()` but no portable device-wide barrier, so a
//! multi-cube sweep would force a host-driven loop (a read-back per sweep). One
//! cube holding `A` + `V` in `SharedMemory` keeps the loop device-resident.
//! v1 sizes (mostly small, one ~256×64) fit one cube (RESEARCH Pattern 2 / A1).
//!
//! ## Layout (one unit per column; A in GLOBAL, V in shared — A1/LDS budget)
//! Unit `c` (`UNIT_POS_X`) owns column `c`. The `m×n` matrix `A` is held
//! COLUMN-MAJOR in the GLOBAL `a_out` handle (`a_out[c*rows + r]`), NOT in shared
//! memory: a `256×64` f32 tile alone is 64 KiB, which together with `V` overflows
//! gfx1100's 64 KiB LDS (RESEARCH A1 / Open Q2 — verified: the all-shared layout
//! requested 82176 > 65536 bytes and the HIP launch was rejected). Keeping `A` in
//! global drops shared usage to `V` (`n×n`, ≤ 16 KiB) + the off-diagonal
//! accumulator, well within budget, while the convergence loop stays in-kernel
//! (the global handle is cube-private for a single-cube launch — D-11 gate 3
//! holds: no HOST round-trip between sweeps). `V` is staged COLUMN-MAJOR in shared
//! (`v_sh[c*MAX_COLS + r]`, initialised to the identity).
//! Within each round-robin step a set of COLUMN-disjoint pairs `(lo, hi)` is
//! enumerated; every unit scans all pair positions for that step and the
//! lower-indexed unit (`c == lo`) of each pair performs the whole rotation
//! (reading both columns from `a_out`/`v_sh`, computing the Jacobi rotation, and
//! writing both back). The `sync_cube()` between steps makes those writes visible
//! before the next step. Because the pairs in a step touch pairwise-disjoint
//! columns, the lo-unit rotations within one step do not write-alias; this is a
//! correctness property of the schedule, NOT a claim that the cpu backend runs the
//! lo units in parallel (the cpu runtime serializes a cube's units — 03-03).
//!
//! ## Round-robin (chess-tournament) pair schedule with ghost padding (CR-01)
//! One sweep must visit ALL `n(n-1)/2` column pairs. The textbook circle method
//! enumerates every pair only when the player count is EVEN; with an ODD player
//! count `n-1` rounds over `n-1` rotating positions silently omit ~half the pairs
//! (`cols=5` visits 6/10, `cols=7` visits 12/21). We therefore pad to an EVEN
//! player count `players = cols` (even) or `cols + 1` (odd — the extra "ghost"
//! player sits out one real pair per round, i.e. the bye), run `players - 1`
//! rounds of `players / 2` positions via the circle method, and SKIP any pairing
//! that touches the ghost column (`hi >= cols`). This visits every real pair
//! exactly once for BOTH parities (verified: cols=4→6/6, 5→10/10, 6→15/15,
//! 7→21/21). Position 0 is the fixed pivot; positions `1..players` rotate.
//!
//! ## Convergence (D-12 constants — RECORDED here)
//! TWO distinct thresholds are passed by value (the key Pitfall-5 fix):
//!   - `skip_thr` — the per-pair rotation-skip bound. A pair `(i, j)` rotates
//!     only when `|γ_ij| > skip_thr`. This MUST stay TINY (`ε_F · ‖A‖_F`) so
//!     rotations are essentially never skipped; conflating it with the break
//!     bound stalls convergence (a loose skip bound stops zeroing real
//!     off-diagonals and the norm plateaus high).
//!   - `conv_thr` — the convergence-break bound. After each full sweep the
//!     off-diagonal Frobenius norm is measured from a CLEAN POST-SWEEP state
//!     (WR-01, mirroring `jacobi_eig.rs`): in a dedicated pass each unit `c`
//!     recomputes the Gram off-diagonals `γ_cj = colᶜ·colʲ` against every other
//!     column `j != c` of the rotated `a_out` and sums `γ_cj²` into `off_sh[c]`,
//!     then a `reduce_sumsq_shared` log₂-tree idiom (in-kernel) reduces them. The
//!     per-column sums double-count each pair (`γ_ij²` appears in column i AND
//!     column j), so `off_sh[0]` holds `2·Σ_{i<j} γ_ij²` and its sqrt is
//!     `sqrt(2)·‖offdiag‖`; the loop breaks when that `≤ conv_thr`. Measuring AFTER
//!     the sweep (not accumulating γ² DURING rotation, which mixes pre/mid-sweep
//!     column states and can declare convergence one sweep early — the exact
//!     pitfall the eig kernel calls out) makes the reported `info[1]` residual
//!     describe the RETURNED matrix, so the host's `NotConverged` guard is sound.
//!     It is set to `8 · ε_F · ‖A‖_F · sqrt(pairs)`
//!     (`pairs = n(n-1)/2`) to account for the ACCUMULATED f32 rounding floor of
//!     the dot products — a single-`ε` bound is unreachable in f32 for moderate
//!     `n` and the loop would hit the cap every time (Pitfall 5). `ε_f32 ≈
//!     1.2e-7`, `ε_f64 ≈ 2.2e-16`.
//! The loop also stops at `max_sweeps = 30` (generous; cyclic one-sided Jacobi
//! converges quadratically). **Forcing case:** the moderate 256×64 case
//! (`svd_moderate_256x64`) plateaus at an off-diagonal norm ≈ `4e-4` (the f32
//! noise floor) while its reconstruction is already within 1e-5; the
//! noise-floor-aware `conv_thr` lets it converge in ~7 sweeps instead of looping
//! to the cap. cuSolver's Jacobi default is `n_iterations = 15`.
//!
//! ## CubeCL expression notes
//! - `SharedMemory::<F>::new(N)` requires a COMPILE-TIME size — `v_sh` is sized to
//!   the comptime cap (`MAX_COLS` × `MAX_COLS`) and the active region is bounded
//!   by the runtime `cols` (mirrors `reduce.rs` sizing 256 + `input.len()` guard).
//!   `A` is NOT in shared (it lives in global `a_out`), so the LDS footprint is
//!   independent of `MAX_ROWS`.
//! - `continue` is NOT supported in `#[cube]` — the "skip below-threshold pair"
//!   is `if gamma.abs() > thr { ...rotate... }` (RESEARCH Pattern 6).
//! - generic constants via `F::from_int` / `F::new`; `Float` methods `.abs()` /
//!   `.sqrt()`.
//! - No hardcoded plane width / 32 — the off-diagonal norm uses the shared-memory
//!   tree (not a plane path), avoiding plane-width portability concerns (D-03).
//!
//! Generic over `<F: Float + CubeElement>` and carries NO backend feature (D-13).
//! Per AGENTS.md §2 this file has NO in-source `mod tests` — the live launch
//! tests are in `crates/mlrs-backend/tests/svd_test.rs`.

use cubecl::prelude::*;

/// Host-side row cap for the tall dimension. `A` lives in global memory so this
/// is NOT a shared-memory size; it bounds the supported problem size and the
/// host (`prims/svd.rs`) rejects `max(rows,cols) > MAX_ROWS` before launch.
pub const MAX_ROWS: u32 = 256;

/// Comptime column cap for the shared `V` tile (`MAX_COLS × MAX_COLS`) and the
/// off-diagonal accumulator. The thin dimension `cols ≤ MAX_COLS`. At f32 this is
/// `64·64·4 = 16 KiB` for `V` + `64·4 = 256 B` for the accumulator — well within
/// gfx1100's 64 KiB LDS (A1).
pub const MAX_COLS: u32 = 64;

/// One-sided Jacobi SVD sweep over a tall `A` (`rows × cols`, `rows ≥ cols`),
/// staged column-major in shared memory. Writes the rotated `A` (column-major,
/// = `U·diag(S)` unnormalized) to `a_out` and the accumulated `V` (column-major,
/// `cols × cols`) to `v_out`. `info_out[0]` carries the sweep count actually run
/// (so the host can detect a non-converged cap hit, D-12 / `NotConverged`).
///
/// - `a_in` is the row-major `rows × cols` input (`a_in[r*cols + c]`).
/// - `a_out` is the rotated matrix in COLUMN-major layout (`a_out[c*rows + r]`).
/// - `v_out` is `V` in COLUMN-major layout (`v_out[c*cols + r]`).
/// - `rows` / `cols` are the runtime active dimensions (`cols ≤ MAX_COLS`,
///   `rows ≤ MAX_ROWS`).
/// - `skip_thr` is the TINY per-pair rotation-skip bound (`ε_F·‖A‖_F`).
/// - `conv_thr` is the off-diagonal-norm convergence-break bound
///   (`8·ε_F·‖A‖_F·sqrt(pairs)`).
/// - `max_sweeps` is the sweep cap (D-12).
///
/// Launch with ONE cube of `cols` units (`CubeDim { x: cols, .. }`).
#[cube(launch)]
pub fn jacobi_svd_sweep<F: Float + CubeElement>(
    a_in: &Array<F>,
    a_out: &mut Array<F>,
    v_out: &mut Array<F>,
    info_out: &mut Array<F>,
    rows: u32,
    cols: u32,
    skip_thr: F,
    conv_thr: F,
    max_sweeps: u32,
) {
    // V staged column-major in shared (v_sh[c*MAX_COLS + r]); A lives in the
    // GLOBAL a_out handle column-major (a_out[c*rows + r]) — A1/LDS budget.
    let mut v_sh = SharedMemory::<F>::new((MAX_COLS * MAX_COLS) as usize);
    // Per-pair off-diagonal contributions γ_ij² for the convergence reduction.
    // One slot per column (the lower-indexed unit of a pair writes its γ²).
    let mut off_sh = SharedMemory::<F>::new(MAX_COLS as usize);

    let c = UNIT_POS_X;
    let zero = F::from_int(0i64);
    let one = F::from_int(1i64);

    // --- Stage: copy my column c of A (row-major in → column-major GLOBAL a_out)
    //     and initialise my column c of V to the identity in shared. Only the
    //     active region (c < cols) participates; OOB units idle. ---
    if c < cols {
        let mut r = 0u32;
        while r < rows {
            // a_in is row-major (rows, cols): element (r, c) at r*cols + c;
            // a_out is column-major: element (r, c) at c*rows + r.
            a_out[(c * rows + r) as usize] = a_in[(r * cols + c) as usize];
            r += 1u32;
        }
        let mut k = 0u32;
        while k < cols {
            let val = if k == c { one } else { zero };
            v_sh[(c * MAX_COLS + k) as usize] = val;
            k += 1u32;
        }
    }
    sync_cube();

    // --- Sweep loop (in-kernel; no host round-trip — D-11 gate 3). ---
    let mut sweep = 0u32;
    let mut converged = false;
    while sweep < max_sweeps && !converged {
        // Ghost-padded round-robin (CR-01): pad to an EVEN player count so the
        // circle method enumerates ALL n(n-1)/2 pairs for odd AND even `cols`.
        // `players = cols` when cols is even, else `cols + 1` (the ghost player
        // >= cols supplies the bye). Run `players - 1` steps of `players/2`
        // positions; SKIP any pairing touching the ghost column (`hi >= cols`).
        let mut players = cols;
        if cols % 2u32 != 0u32 {
            players = cols + 1u32;
        }
        let mut n_steps = 0u32;
        if players > 0u32 {
            n_steps = players - 1u32;
        }
        let mut step = 0u32;
        while step < n_steps {
            // Circle-method pairing for this step over `players` positions: player
            // 0 is fixed, positions 1..players rotate. Pair position p with
            // position (players - 1 - p) for p in [0, players/2). Every unit scans
            // all pair positions and the lo-unit (`c == lo`) of each real pair acts
            // (the pairs in a step are column-disjoint, so the acting units do not
            // write-alias within the step).
            if c < cols {
                let half = players / 2u32;
                let mut p = 0u32;
                while p < half {
                    // position -> player column under the circle rotation (over the
                    // padded `players`); a player >= cols is the ghost.
                    let col_a = circle_player(p, step, players);
                    let col_b = circle_player(players - 1u32 - p, step, players);
                    // Order the pair so (lo, hi).
                    let lo = if col_a < col_b { col_a } else { col_b };
                    let hi = if col_a < col_b { col_b } else { col_a };
                    // Only the LOW-column unit performs this pair's rotation, and
                    // only for REAL pairs (skip ghost pairs `hi >= cols` and the
                    // self-pair `lo == hi`).
                    if c == lo && lo != hi && hi < cols {
                        // --- α = Σ a_ki², β = Σ a_kj², γ = Σ a_ki·a_kj. ---
                        let mut alpha = zero;
                        let mut beta = zero;
                        let mut gamma = zero;
                        let mut r = 0u32;
                        while r < rows {
                            let aki = a_out[(lo * rows + r) as usize];
                            let akj = a_out[(hi * rows + r) as usize];
                            alpha += aki * aki;
                            beta += akj * akj;
                            gamma += aki * akj;
                            r += 1u32;
                        }

                        // --- Jacobi rotation that zeroes γ (skip only when γ is
                        //     below the TINY skip bound; `continue` unsupported →
                        //     if-wrap). The off-diagonal norm is NOT accumulated
                        //     here (WR-01) — it is measured in a clean post-sweep
                        //     pass below so the convergence test reflects the
                        //     RETURNED matrix, not a within-sweep mixture. ---
                        if gamma.abs() > skip_thr {
                            let two = F::from_int(2i64);
                            let zeta = (beta - alpha) / (two * gamma);
                            // t = sign(zeta) / (|zeta| + sqrt(1 + zeta²)).
                            let denom = zeta.abs() + (one + zeta * zeta).sqrt();
                            let mut t = one / denom;
                            // sign(zeta): statement form (no expression-if).
                            if zeta < zero {
                                t = -t;
                            }
                            let cs = one / (one + t * t).sqrt();
                            let sn = cs * t;
                            // Apply to columns lo, hi of A (global) and V (shared).
                            let mut rr = 0u32;
                            while rr < rows {
                                let aki = a_out[(lo * rows + rr) as usize];
                                let akj = a_out[(hi * rows + rr) as usize];
                                a_out[(lo * rows + rr) as usize] = cs * aki - sn * akj;
                                a_out[(hi * rows + rr) as usize] = sn * aki + cs * akj;
                                rr += 1u32;
                            }
                            let mut kk = 0u32;
                            while kk < cols {
                                let vki = v_sh[(lo * MAX_COLS + kk) as usize];
                                let vkj = v_sh[(hi * MAX_COLS + kk) as usize];
                                v_sh[(lo * MAX_COLS + kk) as usize] = cs * vki - sn * vkj;
                                v_sh[(hi * MAX_COLS + kk) as usize] = sn * vki + cs * vkj;
                                kk += 1u32;
                            }
                        }
                    }
                    p += 1u32;
                }
            }
            sync_cube();
            step += 1u32;
        }

        // --- In-kernel off-diagonal-norm convergence test (no host round-trip).
        //     CLEAN POST-SWEEP MEASUREMENT (WR-01, mirrors jacobi_eig.rs): now that
        //     this sweep's rotations are complete, unit `c` recomputes the Gram
        //     off-diagonals γ_cj = colᶜ·colʲ against every other column j != c of
        //     the rotated `a_out` and sums γ_cj² into off_sh[c]. This is a real
        //     measurement of where the matrix stands AFTER the sweep — not an
        //     in-sweep estimate a later rotation could refill. The per-column sums
        //     double-count each pair, so off_sh[0] holds 2·Σ_{i<j} γ_ij² (the
        //     sqrt(2) is folded into the host's conv_thr comparison the same way as
        //     eig — it only makes the break marginally stricter, which is safe). ---
        if c < cols {
            let mut acc = zero;
            let mut j = 0u32;
            while j < cols {
                if j != c {
                    let mut gamma = zero;
                    let mut r = 0u32;
                    while r < rows {
                        let aci = a_out[(c * rows + r) as usize];
                        let acj = a_out[(j * rows + r) as usize];
                        gamma += aci * acj;
                        r += 1u32;
                    }
                    acc += gamma * gamma;
                }
                j += 1u32;
            }
            off_sh[c as usize] = acc;
        }
        sync_cube();

        // Tree-reduce off_sh[0..cols] (= 2·Σ_{i<j} γ_ij²) into off_sh[0].
        let mut s = next_pow2_half(cols);
        while s > 0u32 {
            if c < s && (c + s) < cols {
                let val = off_sh[(c + s) as usize];
                off_sh[c as usize] += val;
            }
            sync_cube();
            s /= 2u32;
        }
        // off_sh[0] now holds 2·Σ_{i<j} γ_ij². Convergence when its sqrt
        // (= sqrt(2)·‖offdiag‖) <= conv_thr.
        let off_norm = off_sh[0usize].sqrt();
        if off_norm <= conv_thr {
            converged = true;
        }
        sync_cube();

        sweep += 1u32;
    }

    // --- Write back: rotated A is ALREADY in a_out (global, in place); write V
    //     (column-major) from shared, plus the sweep count + final norm so the
    //     host can detect a cap hit (NotConverged). ---
    if c < cols {
        let mut k = 0u32;
        while k < cols {
            v_out[(c * cols + k) as usize] = v_sh[(c * MAX_COLS + k) as usize];
            k += 1u32;
        }
    }
    if c == 0u32 {
        // info_out[0] = sweeps run; info_out[1] = final off-diagonal norm.
        info_out[0usize] = F::cast_from(sweep);
        info_out[1usize] = off_sh[0usize].sqrt();
    }
}

/// Circle-method player column for circle position `pos` at rotation `step`,
/// over `players` positions (the EVEN-padded count — `cols` or `cols+1`, CR-01).
/// Position 0 is the fixed pivot (player 0); positions `1..players` rotate:
/// player = ((pos - 1 + step) mod (players - 1)) + 1. With an even `players` the
/// standard round-robin tournament covers all `players·(players-1)/2` position
/// pairs over `players - 1` steps; the caller skips any pair touching a ghost
/// position (`>= cols`), leaving exactly the `cols·(cols-1)/2` real pairs.
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
