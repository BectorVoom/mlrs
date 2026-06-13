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
