//! Cholesky factorization + forward/back triangular solve kernel (D-02) — a
//! single-cube, all-shared-memory `#[cube(launch)]` routine that factors a small
//! SPD matrix `A` (`n×n`, `n ≤ MAX_DIM = 64`) as `A = L·Lᵀ` and solves the dense
//! system `A·x = b` for one or more right-hand-side columns ENTIRELY in-kernel
//! (factor → forward solve → back solve), with NO host round-trip between the
//! three phases (D-11 gate 3). It is the single genuinely-new device primitive of
//! Phase 4: Ridge's normal-equations solve `(XᵀX + αI)·coef = Xᵀy` needs an SPD
//! solve that has no Phase-2/3 analogue (D-02).
//!
//! ## This is the [`crate::jacobi_eig`] blueprint, not a new pattern
//! The order is tiny (`A` is the `n×n` covariance/Gram with `n ≤ MAX_DIM`), so —
//! exactly like the symmetric-eig kernel — the lower factor `L` fits comfortably
//! in shared memory (`MAX_DIM² · 4 B = 16 KiB` at the f32 cap, `32 KiB` at f64,
//! both within gfx1100's 64 KiB LDS). Every factor/solve read/write is therefore
//! LDS-resident. Symmetry of `A` is TRUSTED (D-06): the kernel only reads the
//! lower triangle it needs and never forms `(A + Aᵀ)/2`; the host validates
//! squareness and feeds the symmetric-by-construction Gram.
//!
//! ## Unit-0-does-all serial schedule (RESEARCH Open Q2)
//! Because `n ≤ 64` makes a fully-serialized factor+solve cheap, and because the
//! Cholesky-Banachiewicz recurrence is inherently sequential (row `i` depends on
//! all earlier rows, each `L[i][j]` on `L[j][j]` and the running dot product),
//! the simplest CORRECT schedule is the eig kernel's "acting unit does the whole
//! operation" idiom: unit 0 performs the entire factorization and both triangular
//! solves while all other units idle, with a `sync_cube()` between phases so the
//! shared `L` is visible cube-wide. This sidesteps the cross-unit shared-memory
//! aliasing a distributed triangular update would create and is proven on the CPU
//! backend by the eig kernel.
//!
//! ## Three in-kernel phases (`sync_cube()` between each)
//! 1. **Cholesky-Banachiewicz factor** (row by row): for `i in 0..n`, `j in 0..=i`
//!    `L[i][j] = (A[i][j] − Σ_{k<j} L[i][k]·L[j][k]) / L[j][j]` for `j < i`, and
//!    `L[i][i] = sqrt(A[i][i] − Σ_{k<i} L[i][k]²)`. The diagonal sqrt argument is
//!    GUARDED: if it falls `≤ NEAR_ZERO_FLOOR` the matrix is not SPD, so the
//!    kernel writes a NEGATIVE flag + the offending pivot index/value into
//!    `info_out` and does NOT emit `√(negative) = NaN` (RESEARCH Pitfall 4). Each
//!    computed `L[i][j]` is written into BOTH the shared tile (for the solve
//!    phases) AND `l_out[i*n + j]` (so the host can check `‖L·Lᵀ − A‖` without
//!    re-deriving the factor). The strictly-upper entries of `l_out` are left 0.
//! 2. **Forward solve** `L·z = b` (per rhs column `c`): for `i in 0..n`,
//!    `z[i] = (b[i] − Σ_{k<i} L[i][k]·z[k]) / L[i][i]`. `z` is staged in `x_out`.
//! 3. **Back solve** `Lᵀ·x = z` (per rhs column `c`): for `i` descending,
//!    `x[i] = (z[i] − Σ_{k>i} L[k][i]·x[k]) / L[i][i]`, written into `x_out`.
//!
//! ## CubeCL expression notes (copied from [`crate::jacobi_eig`])
//! - `SharedMemory::<F>::new(N)` requires a COMPILE-TIME size — `l_sh` is sized to
//!   the comptime cap (`MAX_DIM × MAX_DIM`) and the active region is bounded by
//!   the runtime `n` (mirrors `reduce.rs` sizing + `len` guard).
//! - `continue` is NOT supported in `#[cube]` — the non-SPD "skip the rest"
//!   branch is `if`-wrapped, never `continue` (RESEARCH Pattern 6).
//! - generic constants via `F::from_int` / `F::new`; `Float` methods `.sqrt()` /
//!   `.abs()`.
//! - NO hardcoded plane width / 32 — the factor/solve use the shared-memory tile,
//!   not a plane path (carried no-hardcoded-plane-width rule).
//!
//! Generic over `<F: Float + CubeElement>` and carries NO backend feature (D-13).
//! Per AGENTS.md §2 this file has NO in-source `mod tests` — the live launch tests
//! are in `crates/mlrs-backend/tests/cholesky_test.rs`.

use cubecl::prelude::*;

use crate::jacobi_eig::MAX_DIM;

/// Cholesky factor + forward/back triangular solve of a square SPD `A` (`n × n`,
/// row-major, TRUSTED symmetric — D-06), staged in shared memory. Factors
/// `A = L·Lᵀ` then solves `A·x = b` for each of the `rhs` right-hand-side columns
/// fully in-kernel (no host round-trip — D-11 gate 3).
///
/// - `a_in` is the row-major `n × n` SPD input (`a_in[r*n + c]`); symmetry is
///   TRUSTED (no `(A+Aᵀ)/2` — D-06).
/// - `b_in` is the row-major `n × rhs` right-hand side (`b_in[r*rhs + c]`).
/// - `x_out` is the row-major `n × rhs` solution (`x_out[r*rhs + c]`); it also
///   stages the forward-solve `z` in place before the back solve overwrites it.
/// - `l_out` is the row-major `n × n` LOWER Cholesky factor (`l_out[i*n + j]`,
///   strictly-upper entries left 0) — exposed EXPLICITLY so the host can check the
///   `‖L·Lᵀ − A‖` invariant without re-deriving the factor.
/// - `info_out` is length 3: `[0] = non-SPD flag` (`< 0` ⇒ a non-positive pivot
///   was hit; `≥ 0` ⇒ SPD/OK), `[1] = pivot index` (the diagonal index where the
///   factorization failed, encoded as a float), and `[2] = pivot value` (the
///   actual non-positive `√` argument, for host diagnosis). For an SPD input all
///   three stay 0.
/// - `n` is the runtime active dimension (`n ≤ MAX_DIM`).
/// - `rhs` is the number of right-hand-side columns.
///
/// Launch with ONE cube of `n` units (`CubeDim { x: n, .. }`).
#[cube(launch)]
pub fn cholesky_solve<F: Float + CubeElement>(
    a_in: &Array<F>,
    b_in: &Array<F>,
    x_out: &mut Array<F>,
    l_out: &mut Array<F>,
    info_out: &mut Array<F>,
    n: u32,
    rhs: u32,
) {
    // L staged row-major in shared (l_sh[r*MAX_DIM + c]); the matrix is small so
    // it fits LDS comfortably (mirrors the eig kernel's a_sh staging).
    let mut l_sh = SharedMemory::<F>::new((MAX_DIM * MAX_DIM) as usize);

    let unit = UNIT_POS_X;
    let zero = F::from_int(0i64);

    // Near-zero floor for the diagonal sqrt argument. A mathematically-SPD matrix
    // can still produce a slightly-negative `A[i][i] − Σ L[i][k]²` under f32
    // cancellation (RESEARCH Pitfall 4); a pivot at/below this floor is treated as
    // non-SPD and flagged rather than fed to `√` (which would emit NaN).
    let floor = F::new(1e-12_f32);

    // --- Initialise info to the "SPD / OK" sentinel before the acting unit runs
    //     (every unit writes the same constant so there is no race). ---
    if unit == 0u32 {
        info_out[0usize] = zero;
        info_out[1usize] = zero;
        info_out[2usize] = zero;
    }
    sync_cube();

    // The whole factor + both solves are performed by unit 0 (the eig "acting unit
    // does the whole operation" idiom). The recurrence is inherently sequential
    // (row i depends on all earlier rows), n ≤ 64 makes serialization cheap, and a
    // single acting unit sidesteps cross-unit shared-memory aliasing.
    if unit == 0u32 {
        // ---- Phase 1: Cholesky-Banachiewicz factorization A = L·Lᵀ. ----
        let mut spd_ok = true;
        let mut i = 0u32;
        while i < n {
            // Off-diagonal entries j < i: L[i][j] = (A[i][j] − Σ_{k<j} L[i][k]·L[j][k]) / L[j][j].
            let mut j = 0u32;
            while j < i {
                let mut sum = a_in[(i * n + j) as usize];
                let mut k = 0u32;
                while k < j {
                    let l_ik = l_sh[(i * MAX_DIM + k) as usize];
                    let l_jk = l_sh[(j * MAX_DIM + k) as usize];
                    sum -= l_ik * l_jk;
                    k += 1u32;
                }
                let l_jj = l_sh[(j * MAX_DIM + j) as usize];
                let val = sum / l_jj;
                l_sh[(i * MAX_DIM + j) as usize] = val;
                l_out[(i * n + j) as usize] = val;
                j += 1u32;
            }

            // Diagonal entry: L[i][i] = sqrt(A[i][i] − Σ_{k<i} L[i][k]²), GUARDED.
            let mut diag = a_in[(i * n + i) as usize];
            let mut k = 0u32;
            while k < i {
                let l_ik = l_sh[(i * MAX_DIM + k) as usize];
                diag -= l_ik * l_ik;
                k += 1u32;
            }
            // GUARD (RESEARCH Pitfall 4): a non-positive sqrt argument means the
            // matrix is not SPD. Flag it (negated pivot value + index) and DO NOT
            // emit NaN. `continue` is unsupported in #[cube] → if-wrap so the rest
            // of the factor writes a safe placeholder instead of √(negative).
            if diag <= floor && spd_ok {
                // Strictly-negative flag (-1), the failing diagonal index, and the
                // actual non-positive pivot value — all unambiguous for the host.
                info_out[0usize] = F::from_int(-1i64);
                info_out[1usize] = F::cast_from(i);
                info_out[2usize] = diag;
                spd_ok = false;
            }
            if spd_ok {
                let l_ii = diag.sqrt();
                l_sh[(i * MAX_DIM + i) as usize] = l_ii;
                l_out[(i * n + i) as usize] = l_ii;
            } else {
                // Non-SPD: write a safe non-zero placeholder so later divisions do
                // not produce NaN/Inf; the host rejects the whole result via the
                // info flag before reading x.
                l_sh[(i * MAX_DIM + i) as usize] = F::new(1.0_f32);
                l_out[(i * n + i) as usize] = zero;
            }
            i += 1u32;
        }

        // ---- Phases 2 + 3: solve A·x = b per rhs column (only when SPD). ----
        if spd_ok {
            let mut c = 0u32;
            while c < rhs {
                // Phase 2: forward solve L·z = b. Stage z in x_out (row-major).
                let mut fi = 0u32;
                while fi < n {
                    let mut sum = b_in[(fi * rhs + c) as usize];
                    let mut k = 0u32;
                    while k < fi {
                        let l_ik = l_sh[(fi * MAX_DIM + k) as usize];
                        let z_k = x_out[(k * rhs + c) as usize];
                        sum -= l_ik * z_k;
                        k += 1u32;
                    }
                    let l_ii = l_sh[(fi * MAX_DIM + fi) as usize];
                    x_out[(fi * rhs + c) as usize] = sum / l_ii;
                    fi += 1u32;
                }

                // Phase 3: back solve Lᵀ·x = z (descending i). x overwrites z in
                // x_out. We iterate i from n-1 down to 0 with an unsigned counter.
                let mut step = 0u32;
                while step < n {
                    let bi = n - 1u32 - step;
                    let mut sum = x_out[(bi * rhs + c) as usize];
                    // Σ_{k>i} L[k][i]·x[k]  (Lᵀ[i][k] = L[k][i]).
                    let mut k = bi + 1u32;
                    while k < n {
                        let l_ki = l_sh[(k * MAX_DIM + bi) as usize];
                        let x_k = x_out[(k * rhs + c) as usize];
                        sum -= l_ki * x_k;
                        k += 1u32;
                    }
                    let l_ii = l_sh[(bi * MAX_DIM + bi) as usize];
                    x_out[(bi * rhs + c) as usize] = sum / l_ii;
                    step += 1u32;
                }
                c += 1u32;
            }
        }
    }
    sync_cube();
}

// tests live in crates/mlrs-backend/tests/cholesky_test.rs
