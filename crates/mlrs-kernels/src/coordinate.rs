//! `coordinate` — coordinate-descent soft-threshold + residual axpy kernels
//! (LINEAR-03/04, D-03 shared CD kernel for Lasso + ElasticNet).
//!
//! Two feature-free `#[cube]` kernels carry the n-heavy device work of ONE
//! coordinate update of sklearn's `enet_coordinate_descent` (`_cd_fast.pyx`,
//! de-screened); the host owns the cyclic loop and the (one-scalar) soft-threshold
//! + duality-gap math (D-10):
//!
//! 1. [`col_dot`] computes the column dot `t_dot = Σ_i X[i*cols + j]·R[i]` — the
//!    `n`-heavy reduction the host fuses with `w_j_old·‖X_j‖²` to form
//!    `t = X_j·R + w_j_old·‖X_j‖²` (RESEARCH §"Code Examples"). The soft-threshold
//!    `w_j = sign(t)·max(|t| − l1_reg, 0)/(norm2_cols[j] + l2_reg)` is ONE scalar,
//!    so the host does it (no device launch for a single value).
//! 2. [`residual_axpy`] applies `R[i] += factor·X[i*cols + j]` over the `n` rows
//!    when the coefficient changed (`factor = w_j_old − w_j_new`) — the residual
//!    update, the `scale`-shaped per-element map (`elementwise.rs`) specialised to
//!    one strided column.
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! The cpu(f64) backend's MLIR lowering rejects `SharedMemory` + mutable `bool`
//! flags + `F::INFINITY` consts + descending-shift loops (plan 05-02 hit this).
//! Both kernels here use ONLY `F`/`u32` accumulators and `if`-guarded forward
//! loops — no `SharedMemory`, no `bool`, no infinity sentinel, no
//! atomics/scatter (the dot GATHERS into one accumulator on unit 0). The column
//! dot runs on unit 0 of a single cube (`n` is the CD sample count, modest);
//! `residual_axpy` is the standard over-provisioned per-element map
//! (`ABSOLUTE_POS`, bounds-checked).
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). The Wave-0 scaffold owns the `pub mod coordinate;` line in
//! `lib.rs`; this file adds its own `pub use` (file-disjoint, parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/cd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::col_dot as cd_col_dot;
pub use self::enet_gap as cd_enet_gap;
pub use self::residual_axpy as cd_residual_axpy;

/// Column dot `t_dot[0] = Σ_i x[i*cols + j]·r[i]` over the `rows` entries of
/// column `j` of a `rows × cols` row-major matrix `x` and the length-`rows`
/// residual `r` (D-03 / RESEARCH §"Code Examples").
///
/// - `x` is the row-major `rows × cols` design matrix.
/// - `r` is the length-`rows` residual.
/// - `out` is a length-1 device array receiving the dot (`out[0]`).
/// - `rows`, `cols`, `j` are scalar args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg`, mirroring `dist_combine_clamp`'s `rows: u32`).
///
/// Launched as a SINGLE cube; only unit 0 GATHERS the running sum into `out[0]`
/// (no atomics / no cross-unit scatter — the cubecl-cpu lowering does not lower
/// cross-unit atomics; plan 05-02). The CD sample count `rows` is modest, so the
/// single-unit accumulation is acceptable and keeps the tie-free reduction order
/// identical to numpy's left-to-right `dot`.
#[cube(launch)]
pub fn col_dot<F: Float + CubeElement>(
    x: &Array<F>,
    r: &Array<F>,
    out: &mut Array<F>,
    rows: u32,
    cols: u32,
    j: u32,
) {
    // Only unit 0 acts (single-cube GATHER — no atomics, no SharedMemory).
    if UNIT_POS == 0 {
        let mut acc = F::from_int(0i64);
        let mut i = 0u32;
        // Forward accumulation (no descending shift, no bool flag, no infinity).
        while i < rows {
            let idx = (i * cols + j) as usize;
            acc += x[idx] * r[i as usize];
            i += 1u32;
        }
        out[0] = acc;
    }
}

/// Residual axpy `r[i] += factor·x[i*cols + j]` over the `rows` entries of column
/// `j` (the CD residual update `R += (w_j_old − w_j_new)·X[:,j]`, RESEARCH
/// §"Code Examples"; `factor = w_j_old − w_j_new`).
///
/// - `x` is the row-major `rows × cols` design matrix.
/// - `r` is the length-`rows` residual, updated IN PLACE.
/// - `factor` is the scalar `F` axpy multiplier passed BY VALUE (A6 — like
///   `elementwise::scale`'s `factor`).
/// - `rows`, `cols`, `j` are scalar args passed by value.
///
/// One unit handles one row at `ABSOLUTE_POS`; bounds-checked on `ABSOLUTE_POS <
/// rows` so the ceiling-division launch may over-provision threads safely
/// (T-0203-01 precedent). No `SharedMemory`, no `bool`, no infinity sentinel.
#[cube(launch)]
pub fn residual_axpy<F: Float + CubeElement>(
    x: &Array<F>,
    r: &mut Array<F>,
    factor: F,
    rows: u32,
    cols: u32,
    j: u32,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize {
        let idx = i * cols as usize + j as usize;
        r[i] += factor * x[idx];
    }
}

/// ElasticNet duality gap (formulation A, `alpha > 0`) computed device-side into
/// ONE scalar `gap_out[0]` (D-10: exactly one scalar gap readback per outer
/// convergence check — `_cd_fast.pyx` `enet_coordinate_descent`).
///
/// Given the `rows × cols` design `x`, residual `r` (length `rows`), target `y`
/// (length `rows`), coefficient `w` (length `cols`), the per-column squared norms
/// `norm2_cols` (length `cols`), and the penalties `l1_reg` (= sklearn `alpha`,
/// the un-normalized L1 term) and `l2_reg` (= sklearn `beta`), this reproduces:
///
/// ```text
/// XtA[j]        = (X.T @ R)[j] - l2_reg * w[j]
/// dual_norm_XtA = max_j |XtA[j]|
/// R_norm2       = R·R ;  w_norm2 = w·w ;  l1_norm = ||w||_1 ;  Ry = R·y
/// if dual_norm_XtA > l1_reg:  const = l1_reg / dual_norm_XtA ;  gap = 0.5*(R_norm2 + R_norm2*const^2)
/// else:                       const = 1 ;                       gap = R_norm2
/// gap += l1_reg*l1_norm - const*Ry + 0.5*l2_reg*(1+const^2)*w_norm2
/// ```
///
/// All of `X.T@R`, the dots, the max, and the assembly run on unit 0 of a single
/// cube (cols/rows are the modest CD feature/sample counts), so NOTHING but the
/// final scalar crosses back to the host — no `SharedMemory`, no atomics, no
/// `bool`, no infinity sentinel (cubecl-cpu MLIR-safe; the dual-norm max is an
/// `if`-guarded forward scan seeded at 0, which is correct since `|XtA[j]| ≥ 0`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn enet_gap<F: Float + CubeElement>(
    x: &Array<F>,
    r: &Array<F>,
    y: &Array<F>,
    w: &Array<F>,
    gap_out: &mut Array<F>,
    rows: u32,
    cols: u32,
    l1_reg: F,
    l2_reg: F,
) {
    if UNIT_POS == 0 {
        let zero = F::from_int(0i64);
        let half = F::new(0.5);
        let one = F::from_int(1i64);

        // R_norm2 = R·R, Ry = R·y (forward dots, numpy left-to-right order).
        let mut r_norm2 = zero;
        let mut ry = zero;
        let mut i = 0u32;
        while i < rows {
            let ri = r[i as usize];
            r_norm2 += ri * ri;
            ry += ri * y[i as usize];
            i += 1u32;
        }

        // dual_norm_XtA = max_j |X.T@R - l2_reg*w|_j ; w_norm2 = w·w ; l1_norm = ||w||_1.
        let mut dual_norm = zero;
        let mut w_norm2 = zero;
        let mut l1_norm = zero;
        let mut j = 0u32;
        while j < cols {
            // (X.T @ R)[j] = Σ_i X[i*cols + j] · R[i].
            let mut xtr = zero;
            let mut ii = 0u32;
            while ii < rows {
                let idx = (ii * cols + j) as usize;
                xtr += x[idx] * r[ii as usize];
                ii += 1u32;
            }
            let wj = w[j as usize];
            let mut xta = xtr - l2_reg * wj;
            // |XtA[j]| via an if-guarded negate (no `abs` intrinsic dependency).
            if xta < zero {
                xta = -xta;
            }
            if xta > dual_norm {
                dual_norm = xta;
            }
            w_norm2 += wj * wj;
            // ||w||_1 += |w[j]|.
            let mut awj = wj;
            if awj < zero {
                awj = -awj;
            }
            l1_norm += awj;
            j += 1u32;
        }

        // const = l1_reg / dual_norm if dual_norm > l1_reg else 1.
        let mut cnst = one;
        if dual_norm > l1_reg {
            cnst = l1_reg / dual_norm;
        }

        // gap = 0.5*(R_norm2 + R_norm2*const^2) when scaled, else R_norm2.
        let mut gap = r_norm2;
        if dual_norm > l1_reg {
            gap = half * (r_norm2 + r_norm2 * cnst * cnst);
        }
        gap += l1_reg * l1_norm - cnst * ry + half * l2_reg * (one + cnst * cnst) * w_norm2;

        gap_out[0] = gap;
    }
}
