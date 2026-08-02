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
//! ## A second kernel for the orders LDS cannot hold
//! Everything above describes [`cholesky_solve`], and every word of it is
//! conditional on `n ≤ MAX_DIM`. `Ridge` at `d = 256` and `KernelRidge` above 64
//! samples are not: a 256-order factor is 256 KiB, which no adapter will stage,
//! and both used to be rejected outright. [`cholesky_solve_wide`] carries
//! `MAX_DIM < n ≤ CHOLESKY_WIDE_MAX_DIM` by keeping `L` in GLOBAL memory,
//! distributing each column over the cube, and staging only `O(n)` in shared.
//! See its own doc comment for the three consequences (and for why it is
//! deliberately not bitwise-equal to the narrow arm).
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
/// - `alpha` is added to the DIAGONAL of `A` as it is read, so the kernel
///   factors `(A + αI)` without anyone materializing that matrix. Pass `0` for
///   the plain `A`. This exists for Ridge: the normal equations are
///   `(XᵀX + αI)·coef = Xᵀy`, and before the parameter existed the caller had to
///   read the whole `d × d` Gram back to the host, add `α` on the diagonal in a
///   loop, and re-upload it — a synchronising round-trip of `2·d²` floats to
///   change `d` of them. Reading `a_in[i*n+i] + alpha` costs nothing.
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
    alpha: F,
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

            // Diagonal entry: L[i][i] = sqrt(A[i][i] + α − Σ_{k<i} L[i][k]²),
            // GUARDED. `alpha` is the ridge penalty, added HERE rather than by a
            // host round-trip over the whole matrix.
            let mut diag = a_in[(i * n + i) as usize] + alpha;
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

/// Largest order [`cholesky_solve_wide`] accepts.
///
/// The wide kernel keeps `L` in GLOBAL memory, so this cap is not a factor
/// footprint the way [`MAX_DIM`] is — it bounds the two length-`n` shared
/// vectors the kernel does stage (`dsum` and the row/`z` scratch), at
/// `2 · 1024 · 8 B = 16 KiB` for `f64` and half that for `f32`. Both sit well
/// inside the 48 KiB every adapter in this codebase reports, and
/// `prims::cholesky` checks the budget rather than assuming it (a shared-memory
/// overrun is SILENT — the `prims::eig` finding).
pub const CHOLESKY_WIDE_MAX_DIM: u32 = 1024;

/// Cholesky factor + solve for an order the shared-memory kernel cannot hold —
/// the `MAX_DIM < n ≤ CHOLESKY_WIDE_MAX_DIM` arm of [`crate::cholesky_solve`].
///
/// Same contract, same `info_out` encoding, same `alpha`-on-the-diagonal
/// convention. Three things differ, and all three follow from `L` no longer
/// fitting in LDS (`n = 256` is a 256 KiB factor at `f32`):
///
/// 1. **`L` lives in `l_out` (global).** Only two length-`n` vectors are staged
///    in shared, so the footprint is `O(n)` rather than `O(n²)`.
/// 2. **The factor is column-parallel, not unit-0-serial.** At `n ≤ 64` a
///    serialized `n³/6` is cheap and the simplest correct schedule wins; at
///    `n = 256` it is 2.8 M dependent operations on ONE lane, which is a
///    millisecond of a fit whose whole Gram costs four. Here the diagonal
///    element of column `j` is a scalar on unit 0 and the `n − j − 1`
///    off-diagonal entries below it are distributed over the cube, so the
///    parallel work is `n³/6` spread over `CUBE_DIM_X` lanes and only the
///    scalar chain is serial.
/// 3. **The serial diagonal chain is removed outright.** `L[j][j]` needs
///    `Σ_{k<j} L[j][k]²`, which is the one part of the recurrence that cannot
///    be distributed at step `j` — but it CAN be accumulated incrementally:
///    every unit that writes `L[i][j]` also folds `L[i][j]²` into `dsum[i]`, so
///    by the time column `j` is reached `dsum[j]` already holds the sum and the
///    diagonal is `O(1)`. Without this the kernel would be `Σj = n²/2`
///    dependent shared-memory reads on unit 0 (~130 µs at `n = 256`), which is
///    the same bottleneck in a different place.
///
/// The two triangular solves are likewise column-oriented (`axpy`) rather than
/// row-oriented (`dot`), for the same reason: an `axpy` sweep is `O(1)` serial
/// plus `(n − k)` parallel work, where the dot form is a length-`i` dependent
/// chain per row.
///
/// **This is NOT bitwise-equal to [`crate::cholesky_solve`]** — the incremental
/// `dsum` and the `axpy` solves re-associate sums the serial kernel evaluates
/// left-to-right. The two arms cover DISJOINT `n` ranges, so no shape can
/// observe both; `cholesky_test.rs` pins the wide arm against an `f64` host
/// oracle and against the narrow arm on an overlapping order (forced via
/// `MLRS_CHOLESKY_WIDE`) within the numerical tolerance, not by `assert_eq`.
///
/// Launch with ONE cube of `min(n, CHOLESKY_WIDE_MAX_DIM)` units.
#[cube(launch)]
pub fn cholesky_solve_wide<F: Float + CubeElement>(
    a_in: &Array<F>,
    b_in: &Array<F>,
    x_out: &mut Array<F>,
    l_out: &mut Array<F>,
    info_out: &mut Array<F>,
    n: u32,
    rhs: u32,
    alpha: F,
) {
    // `dsum[i] = Σ_{k<j} L[i][k]²` at step `j`, maintained incrementally (see
    // the doc comment); `scratch` stages row `j` of `L` during the factor and
    // the forward-solve vector `z` during the solve. Both are length-`n`, sized
    // to the comptime cap and bounded by the runtime `n` (the `reduce.rs`
    // sizing + `len` guard idiom).
    let mut dsum = SharedMemory::<F>::new(CHOLESKY_WIDE_MAX_DIM as usize);
    let mut scratch = SharedMemory::<F>::new(CHOLESKY_WIDE_MAX_DIM as usize);

    let unit = UNIT_POS_X;
    let units = CUBE_DIM_X;
    let zero = F::from_int(0i64);
    let floor = F::new(1e-12_f32);

    // --- Zero `l_out` (the strictly-upper triangle is part of this kernel's
    //     contract, and a pooled buffer is not guaranteed clean) and `dsum`. ---
    let total = n * n;
    let mut z = unit;
    while z < total {
        l_out[z as usize] = zero;
        z += units;
    }
    let mut zi = unit;
    while zi < n {
        dsum[zi as usize] = zero;
        zi += units;
    }
    if unit == 0u32 {
        info_out[0usize] = zero;
        info_out[1usize] = zero;
        info_out[2usize] = zero;
    }
    sync_cube();

    // ---- Phase 1: column-oriented factorization A + αI = L·Lᵀ. ----
    let mut j = 0u32;
    while j < n {
        // Stage row `j` of `L` (entries k < j) so the off-diagonal updates below
        // read it from LDS instead of `n − j − 1` lanes each streaming the same
        // `j` global addresses.
        let mut k = unit;
        while k < j {
            scratch[k as usize] = l_out[(j * n + k) as usize];
            k += units;
        }
        sync_cube();

        // The pivot, in O(1) off the incremental `dsum`. GUARDED exactly as the
        // narrow kernel guards it: a non-positive argument is flagged, never fed
        // to `√`.
        if unit == 0u32 {
            let diag = a_in[(j * n + j) as usize] + alpha - dsum[j as usize];
            if diag <= floor {
                // Record only the FIRST failing pivot, then carry on with a
                // safe unit placeholder. Every barrier below is unconditional,
                // so the kernel MUST NOT branch the rest of the factor (or the
                // solve) on the flag: a WGSL barrier in control flow predicated
                // on a value loaded from a buffer fails the uniformity analysis
                // even when the value is uniform in fact. A finite-garbage
                // factor that the host discards on `info_out[0]` is the cheap
                // way to keep the schedule branch-free.
                if info_out[0usize] >= zero {
                    info_out[0usize] = F::from_int(-1i64);
                    info_out[1usize] = F::cast_from(j);
                    info_out[2usize] = diag;
                }
                l_out[(j * n + j) as usize] = F::new(1.0_f32);
            } else {
                l_out[(j * n + j) as usize] = diag.sqrt();
            }
        }
        sync_cube();

        let l_jj = l_out[(j * n + j) as usize];
        let mut i = j + 1u32 + unit;
        while i < n {
            let mut sum = a_in[(i * n + j) as usize];
            let mut kk = 0u32;
            while kk < j {
                sum -= l_out[(i * n + kk) as usize] * scratch[kk as usize];
                kk += 1u32;
            }
            let val = sum / l_jj;
            l_out[(i * n + j) as usize] = val;
            // Row `i` is owned by exactly one unit within this step, so the
            // accumulate is race-free without an atomic.
            dsum[i as usize] += val * val;
            i += units;
        }
        sync_cube();
        j += 1u32;
    }

    // ---- Phases 2 + 3: solve (A + αI)·x = b per rhs column. ----
    let mut c = 0u32;
    while c < rhs {
        // Phase 2: forward solve L·z = b, column-oriented.
        let mut i0 = unit;
        while i0 < n {
            scratch[i0 as usize] = b_in[(i0 * rhs + c) as usize];
            i0 += units;
        }
        sync_cube();

        let mut fk = 0u32;
        while fk < n {
            if unit == 0u32 {
                scratch[fk as usize] = scratch[fk as usize] / l_out[(fk * n + fk) as usize];
            }
            sync_cube();
            let zk = scratch[fk as usize];
            let mut i = fk + 1u32 + unit;
            while i < n {
                scratch[i as usize] -= l_out[(i * n + fk) as usize] * zk;
                i += units;
            }
            sync_cube();
            fk += 1u32;
        }

        // Phase 3: back solve Lᵀ·x = z, descending, also column-oriented.
        // `Lᵀ[i][k] = L[k][i]`, so the axpy at step `k` reads across row `k`.
        let mut step = 0u32;
        while step < n {
            let bk = n - 1u32 - step;
            if unit == 0u32 {
                let xv = scratch[bk as usize] / l_out[(bk * n + bk) as usize];
                scratch[bk as usize] = xv;
                x_out[(bk * rhs + c) as usize] = xv;
            }
            sync_cube();
            let xk = scratch[bk as usize];
            let mut i = unit;
            while i < bk {
                scratch[i as usize] -= l_out[(bk * n + i) as usize] * xk;
                i += units;
            }
            sync_cube();
            step += 1u32;
        }
        c += 1u32;
    }
}

// tests live in crates/mlrs-backend/tests/cholesky_test.rs
