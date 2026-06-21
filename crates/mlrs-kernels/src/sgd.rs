//! `sgd` — minibatch-SGD margin (pass 1) + weight-update (pass 2) kernels
//! (PRIM-10, SGDSVM-01..02).
//!
//! Two feature-free `#[cube]` kernels carry the n/d-heavy device work of ONE
//! minibatch SGD step; the host owns the per-sample `dloss` table, the learning-
//! rate schedule (`optimal` t0 / `invscaling`), and the intercept update (all
//! f64 scalars). The two-pass split keeps every device write SINGLE-OWNER (no
//! lock-free reductions / no scatter), mirroring the shipped `coordinate.rs`
//! GATHER idiom:
//!
//! 1. [`sgd_margin`] (pass 1, per-sample) computes the linear margin
//!    `p[i] = Σ_j x[i*d + j]·w[j] + bias` over the `b` minibatch rows — the
//!    over-provisioned per-element map (`ABSOLUTE_POS`, bounds-checked), the
//!    `residual_axpy` shape specialised to a full forward dot.
//! 2. [`sgd_weight_update`] (pass 2, per-coordinate) applies the SGD coordinate
//!    update `w[j] -= eta · inv_b · Σ_i g[i]·x[i*d + j]` — one unit per
//!    coordinate `j` GATHERS the minibatch gradient (single-owner reduction, the
//!    `col_dot` shape) and writes its own `w[j]`. The host supplies `g[i] =
//!    dloss(p_i, y_i)` and `eta` (the scheduled step).
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! The cpu(f64) backend's MLIR lowering rejects shared-memory tiles + mutable
//! `bool` flags + the floating infinity constant + descending-shift loops (plan
//! 05-02 hit this). Both kernels here use ONLY `F`/`u32` accumulators and
//! `if`-guarded forward loops — no shared-memory tile, no `bool`, no infinity
//! sentinel, single-owner reductions/no scatter (each GATHERS into one
//! accumulator the owning
//! unit writes). The margin pass is the standard over-provisioned per-element map
//! (`ABSOLUTE_POS`, bounds-checked); the update pass runs one unit per coordinate.
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). The Wave-0 scaffold (plan 10-01) owns the `pub mod sgd;` line
//! in `lib.rs`; this file adds its own `pub use` (file-disjoint, parallel-safe).
//! For Wave-0 the bodies are the real GATHER shape; the host `dloss`/schedule
//! helper + the prim epoch loop that drives these land in Wave-1.
//!
//! Tests live in `crates/mlrs-backend/tests/sgd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::sgd_margin as sgd_margin_kernel;
pub use self::sgd_weight_update as sgd_weight_update_kernel;

/// Pass 1 (per-sample margin): `p[i] = Σ_j x[i*d + j]·w[j] + bias` over the `b`
/// minibatch rows of the row-major `b × d` design `x` and the length-`d` weight
/// vector `w`.
///
/// - `x` is the row-major `b × d` minibatch design matrix.
/// - `w` is the length-`d` weight vector.
/// - `bias` is the scalar intercept passed BY VALUE (A6 — like
///   `residual_axpy`'s `factor`).
/// - `p` is the length-`b` output margin vector (`p[i]`).
/// - `b`, `d` are scalar args passed BY VALUE (cubecl 0.10 — no `ScalarArg`).
///
/// One unit handles one row at `ABSOLUTE_POS`; bounds-checked on `i < b` so the
/// ceiling-division launch may over-provision threads safely. The dot is a
/// forward `while` scan seeded `F::from_int(0i64)` — no shared-memory tile, no
/// `bool`, no infinity sentinel (cubecl-cpu MLIR-safe).
#[cube(launch)]
pub fn sgd_margin<F: Float + CubeElement>(
    x: &Array<F>,
    w: &Array<F>,
    bias: F,
    p: &mut Array<F>,
    b: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < b as usize {
        let mut acc = F::from_int(0i64);
        let mut j = 0u32;
        // Forward accumulation (no descending shift, no bool flag, no infinity).
        while j < d {
            let idx = i * d as usize + j as usize;
            acc += x[idx] * w[j as usize];
            j += 1u32;
        }
        p[i] = acc + bias;
    }
}

/// Pass 2 (per-coordinate weight update): `w[j] -= eta · inv_b · Σ_i g[i]·x[i*d
/// + j]` over the `b` minibatch rows for coordinate `j` of the row-major `b × d`
/// design `x`.
///
/// - `x` is the row-major `b × d` minibatch design matrix.
/// - `g` is the length-`b` per-sample gradient `g[i] = dloss(p_i, y_i)` (host-
///   supplied, the `dloss` table is host f64).
/// - `w` is the length-`d` weight vector, updated IN PLACE (single-owner write
///   per coordinate `j`).
/// - `eta` is the scheduled learning rate passed BY VALUE.
/// - `inv_b` is the `1/batch_size` averaging factor passed BY VALUE (A2 — the
///   minibatch sum-vs-average choice is the host's; the kernel just multiplies).
/// - `d`, `b` are scalar args passed BY VALUE.
///
/// One unit handles one coordinate at `ABSOLUTE_POS`; bounds-checked on `j < d`.
/// The minibatch gradient is a forward `while` GATHER seeded `F::from_int(0i64)`
/// — single-owner (each unit reduces its own column and writes its own `w[j]`),
/// no cross-unit reduction/scatter, no shared-memory tile, no `bool`, no infinity sentinel
/// (cubecl-cpu MLIR-safe).
#[cube(launch)]
pub fn sgd_weight_update<F: Float + CubeElement>(
    x: &Array<F>,
    g: &Array<F>,
    w: &mut Array<F>,
    eta: F,
    inv_b: F,
    d: u32,
    b: u32,
) {
    let j = ABSOLUTE_POS;
    if j < d as usize {
        let mut grad = F::from_int(0i64);
        let mut i = 0u32;
        // Forward GATHER over the minibatch rows (single-owner, no scatter).
        while i < b {
            let idx = i as usize * d as usize + j;
            grad += g[i as usize] * x[idx];
            i += 1u32;
        }
        w[j] -= eta * inv_b * grad;
    }
}
