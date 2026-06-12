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
//! ## Layout (one unit per column)
//! Unit `c` (`UNIT_POS_X`) owns column `c` of the `m×n` matrix `A` and column
//! `c` of the `n×n` accumulator `V`. `A` is staged COLUMN-MAJOR in shared memory
//! (`a_sh[c*MAX_ROWS + r]`) so a column is a contiguous owned strip; `V` is
//! likewise column-major (`v_sh[c*MAX_COLS + r]`, initialised to the identity).
//! Per round-robin step a disjoint set of column pairs `(i, j)` rotate
//! concurrently — each pair is handled by the lower-indexed unit, which reads
//! both columns, computes the Jacobi rotation, and writes both back; the
//! `sync_cube()` between steps makes the writes visible before the next step.
//!
//! ## Round-robin (chess-tournament) pair schedule (RESEARCH Pattern 6)
//! One sweep visits all `n(n-1)/2` column pairs over `n-1` steps, `floor(n/2)`
//! disjoint pairs per step. We use the standard circle method: index 0 is fixed,
//! the rest rotate. Disjoint pairs touch disjoint columns, so the rotations
//! within a step are write-conflict-free.
//!
//! ## Convergence (D-12 constants — RECORDED here)
//! After each full sweep the off-diagonal Frobenius norm `sqrt(Σ_{i<j} γ_ij²)`
//! (with `γ_ij = column_i · column_j`) is reduced with the `reduce_sumsq_shared`
//! log₂-tree idiom (in-kernel). The loop breaks when that norm falls below
//! `threshold` or the sweep count reaches `max_sweeps`. The host passes both by
//! value: `threshold = 8 · ε_F · ‖A‖_F` (ε_f32 ≈ 1.2e-7, ε_f64 ≈ 2.2e-16) and
//! `max_sweeps = 30`. **Forcing case:** the moderate 256×64 case
//! (`svd_moderate_256x64`) and the clustered/repeated D-08 cases need a generous
//! cap; cyclic one-sided Jacobi converges quadratically (~10–15 sweeps for
//! n ≤ 256, cuSolver's Jacobi default is `n_iterations = 15`), so 30 is generous
//! headroom while the `8·ε_F·‖A‖_F` threshold stays reachable in f32 (a tighter
//! f64-grade threshold would loop to the cap every time in f32 — Pitfall 5).
//!
//! ## CubeCL expression notes
//! - `SharedMemory::<F>::new(N)` requires a COMPILE-TIME size — the tile is sized
//!   to a comptime cap (`MAX_ROWS` × `MAX_COLS`) and the active region is bounded
//!   by the runtime `(rows, cols)` (mirrors `reduce.rs` sizing 256 + `input.len()`
//!   guard).
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

/// Comptime row cap for the staged `A` tile. The active row count `rows` is a
/// runtime value bounded by this; a launch with `rows > MAX_ROWS` is rejected by
/// the host before launch (`prims/svd.rs`).
pub const MAX_ROWS: u32 = 256;

/// Comptime column cap for the staged `A` / `V` tiles.
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
/// - `threshold` is the off-diagonal-norm convergence bound (`8·ε_F·‖A‖_F`).
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
    threshold: F,
    max_sweeps: u32,
) {
    // Column-major staging: a_sh[c*MAX_ROWS + r], v_sh[c*MAX_COLS + r].
    let mut a_sh = SharedMemory::<F>::new((MAX_ROWS * MAX_COLS) as usize);
    let mut v_sh = SharedMemory::<F>::new((MAX_COLS * MAX_COLS) as usize);
    // Per-pair off-diagonal contributions γ_ij² for the convergence reduction.
    // One slot per column (the lower-indexed unit of a pair writes its γ²).
    let mut off_sh = SharedMemory::<F>::new(MAX_COLS as usize);

    let c = UNIT_POS_X;
    let zero = F::from_int(0i64);
    let one = F::from_int(1i64);

    // --- Stage: load my column c of A (row-major in → column-major shared) and
    //     initialise my column c of V to the identity. Only the active region
    //     (c < cols) participates; OOB units idle. ---
    if c < cols {
        let mut r = 0u32;
        while r < rows {
            // a_in is row-major (rows, cols): element (r, c) at r*cols + c.
            a_sh[(c * MAX_ROWS + r) as usize] = a_in[(r * cols + c) as usize];
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
        // Reset this sweep's per-pair off-diagonal accumulator.
        if c < cols {
            off_sh[c as usize] = zero;
        }
        sync_cube();

        // Round-robin schedule: n-1 steps, each a disjoint set of pairs. We use
        // the circle method — column 0 fixed, the rest rotate. For step s, unit
        // `c` (when it is the LOW member of its pair) rotates its pair.
        // To keep the schedule simple and correct on a single cube we iterate the
        // canonical pair set (i<j) but partition rotations so disjoint pairs in a
        // step run concurrently and a sync_cube separates steps.
        let mut n_steps = 0u32;
        if cols > 0u32 {
            n_steps = cols - 1u32;
        }
        let mut step = 0u32;
        while step < n_steps {
            // Circle-method pairing for this step: position p in [0, cols) maps to
            // a "player". Player 0 is fixed; player at position p (p>=1) is
            // ((p - 1 + step) % (cols - 1)) + 1. Pair position p with position
            // (cols - 1 - p) for p in [0, cols/2).
            // Each unit determines whether it is the LOW column of exactly one
            // pair this step and, if so, rotates it.
            if c < cols {
                // Find c's position in the circle ordering, then its partner.
                // Build the partner column for `c` this step. We compute, for each
                // pair position p, the two player columns and let the lower one act.
                // To avoid per-unit search, every unit scans the pair positions.
                let half = cols / 2u32;
                let mut p = 0u32;
                while p < half {
                    // position -> player column under the circle rotation.
                    let col_a = circle_player(p, step, cols);
                    let col_b = circle_player(cols - 1u32 - p, step, cols);
                    // Order the pair so (lo, hi).
                    let lo = if col_a < col_b { col_a } else { col_b };
                    let hi = if col_a < col_b { col_b } else { col_a };
                    // Only the LOW-column unit performs this pair's rotation.
                    if c == lo && lo != hi {
                        // --- α = Σ a_ki², β = Σ a_kj², γ = Σ a_ki·a_kj. ---
                        let mut alpha = zero;
                        let mut beta = zero;
                        let mut gamma = zero;
                        let mut r = 0u32;
                        while r < rows {
                            let aki = a_sh[(lo * MAX_ROWS + r) as usize];
                            let akj = a_sh[(hi * MAX_ROWS + r) as usize];
                            alpha += aki * aki;
                            beta += akj * akj;
                            gamma += aki * akj;
                            r += 1u32;
                        }
                        // Record this pair's off-diagonal contribution γ² for the
                        // convergence test (accumulate into the lo slot).
                        off_sh[lo as usize] += gamma * gamma;

                        // --- Jacobi rotation that zeroes γ (skip if below thr;
                        //     `continue` unsupported → if-wrap). ---
                        if gamma.abs() > threshold {
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
                            // Apply to columns lo, hi of A and V.
                            let mut rr = 0u32;
                            while rr < rows {
                                let aki = a_sh[(lo * MAX_ROWS + rr) as usize];
                                let akj = a_sh[(hi * MAX_ROWS + rr) as usize];
                                a_sh[(lo * MAX_ROWS + rr) as usize] = cs * aki - sn * akj;
                                a_sh[(hi * MAX_ROWS + rr) as usize] = sn * aki + cs * akj;
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
        //     Tree-reduce off_sh[0..cols] (= Σ_{i<j} γ_ij²) into off_sh[0]. ---
        let mut s = next_pow2_half(cols);
        while s > 0u32 {
            if c < s && (c + s) < cols {
                let val = off_sh[(c + s) as usize];
                off_sh[c as usize] += val;
            }
            sync_cube();
            s /= 2u32;
        }
        // off_sh[0] now holds Σ γ². Convergence when sqrt(Σγ²) <= threshold.
        let off_norm = off_sh[0usize].sqrt();
        if off_norm <= threshold {
            converged = true;
        }
        sync_cube();

        sweep += 1u32;
    }

    // --- Write back: rotated A (column-major) and V (column-major), plus the
    //     sweep count so the host can detect a cap hit (NotConverged). ---
    if c < cols {
        let mut r = 0u32;
        while r < rows {
            a_out[(c * rows + r) as usize] = a_sh[(c * MAX_ROWS + r) as usize];
            r += 1u32;
        }
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
/// over `cols` players. Position 0 is the fixed pivot (player 0); positions
/// `1..cols` rotate: player = ((pos - 1 + step) mod (cols - 1)) + 1. This yields
/// the standard round-robin tournament so one sweep covers all `n(n-1)/2` pairs
/// over `n-1` steps.
#[cube]
fn circle_player(pos: u32, step: u32, cols: u32) -> u32 {
    let mut player = 0u32;
    if pos > 0u32 {
        let m = cols - 1u32;
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
