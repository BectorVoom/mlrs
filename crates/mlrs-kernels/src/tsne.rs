//! t-SNE device kernels (TSNE-01) — the two O(n²) per-iteration passes of the
//! EXACT-method Kullback-Leibler gradient.
//!
//! Per gradient-descent iteration the exact t-SNE objective needs, from the
//! current embedding `y` (`n × d`, row-major):
//!
//! 1. `qnum[i,j] = (1 + ‖y_i − y_j‖²/dof)^(−(dof+1)/2)` (the UNNORMALISED
//!    Student-t affinity; diagonal forced 0) — [`tsne_qnum`], fed by the
//!    Phase-2 `distance(sqrt=false)` prim's squared-distance block.
//! 2. `grad[i,c] = c_f · Σ_j (p_ij − qnum_ij/qsum) · qnum_ij · (y_ic − y_jc)`
//!    with `c_f = 2(dof+1)/dof` (sklearn `_kl_divergence`; `= 4` at the
//!    `n_components = 2` default `dof = 1`) — [`tsne_grad`]. `Q = qnum/qsum`
//!    is clamped below at `eps` (sklearn `MACHINE_EPSILON` — the f64 eps even
//!    on the f32 arm) exactly as `np.maximum(Q, eps)` before the `(P − Q)`
//!    difference.
//!
//! ## cpu-MLIR authoring contract (the `distance.rs`/`mutual_reachability.rs`
//! rules, VALIDATED)
//! - Per-element 2D launch: `ABSOLUTE_POS_X`/`ABSOLUTE_POS_Y` with a 16×16
//!   cube and ceiling-div counts, guarded `if i < rows { if j < cols { … } }`.
//! - STATEMENT-form conditionals only (mutable-`F` `if` guards) — never an
//!   `if`-expression in value position.
//! - Single-loop `F` accumulator only (the `manhattan_dist` idiom) — no
//!   cross-sibling-loop accumulator (FINDING 002-B), no `SharedMemory`, no
//!   `Atomic`, no `F::INFINITY`.
//! - Scalars pass by value (cubecl 0.10).
//!
//! Host-launch wrappers live in `mlrs-backend` (`prims/tsne.rs`, which owns the
//! concrete `ActiveRuntime`); the VALUE oracle is asserted there under a real
//! runtime. This crate stays backend-feature-free (Criterion 1).

use cubecl::prelude::*;

/// Unnormalised Student-t affinity from a squared-distance block:
/// `out[i*n + j] = (1 + dsq[i*n + j] * inv_dof)^exponent` off-diagonal, `0` on
/// the diagonal. `inv_dof = 1/degrees_of_freedom`, `exponent = −(dof+1)/2`
/// (both precomputed host-side; at the `n_components = 2` default they are
/// `1` and `−1`, making this exactly `1/(1 + d²)`).
///
/// One unit per output element `(i, j)`; the diagonal zero is a STATEMENT-form
/// overwrite, never an `if`-expression in value position.
#[cube(launch)]
pub fn tsne_qnum<F: Float + CubeElement>(
    dsq: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    inv_dof: F,
    exponent: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < n {
        if j < n {
            let idx = (i * n + j) as usize;
            let base = F::from_int(1i64) + dsq[idx] * inv_dof;
            let mut v = F::powf(base, exponent);
            if i == j {
                v = F::from_int(0i64);
            }
            out[idx] = v;
        }
    }
}

/// Direct squared-Euclidean pairwise distance: `out[i*n+j] = Σ_k (x[i,k] -
/// x[j,k])²`. One unit per output element `(i, j)`, single-loop `F`
/// accumulator (the `manhattan_dist` idiom) over the (small — `n_features`
/// or `n_components`) feature dimension `d`. Deliberately NOT the
/// GEMM-expansion `distance` prim: that prim's norm term forces
/// `row_reduce(ReducePath::Shared)` (CR-01), whose SharedMemory-barrier
/// emulation on `cubecl-cpu` is pathologically slow in some host-threading
/// contexts (measured: single-digit-ms in a standalone test binary, tens of
/// seconds+ embedded under PyO3) — a direct small-`d` GATHER sidesteps the
/// path entirely and is the right shape for t-SNE's tiny `d` regardless.
#[cube(launch)]
pub fn tsne_sqdist<F: Float + CubeElement>(x: &Array<F>, out: &mut Array<F>, n: u32, d: u32) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < n {
        if j < n {
            let mut acc = F::from_int(0i64);
            let mut k = 0u32;
            while k < d {
                let diff = x[(i * d + k) as usize] - x[(j * d + k) as usize];
                acc += diff * diff;
                k += 1u32;
            }
            out[(i * n + j) as usize] = acc;
        }
    }
}

/// Plain per-row sum GATHER: `out[i] = Σ_j m[i*cols + j]` — one unit per ROW,
/// single-loop `F` accumulator (the `manhattan_dist` idiom). Exists because the
/// generic `row_reduce(ReducePath::Shared)` path emulates `SharedMemory`
/// barriers on the cpu runtime at ~11 s per 48×48 call (measured) — a plain
/// loop is sub-millisecond and needs no cross-thread state.
#[cube(launch)]
pub fn tsne_rowsum<F: Float + CubeElement>(m: &Array<F>, out: &mut Array<F>, rows: u32, cols: u32) {
    let i = ABSOLUTE_POS_X;
    if i < rows {
        let mut acc = F::from_int(0i64);
        let mut j = 0u32;
        while j < cols {
            acc += m[(i * cols + j) as usize];
            j += 1u32;
        }
        out[i as usize] = acc;
    }
}

/// Exact-method KL gradient GATHER:
/// `out[i*d + c] = c_f · Σ_j (p[i,j] − max(qnum[i,j] · inv_qsum, eps)) ·
///                 qnum[i,j] · (y[i,c] − y[j,c])`.
///
/// One unit per gradient element `(i, c)` (`i` on `ABSOLUTE_POS_X`, `c` on
/// `ABSOLUTE_POS_Y` — `d` is tiny, so the launch is effectively 1D over `i`).
/// The `j` loop is a single-loop `F` accumulator (the `manhattan_dist` idiom).
/// The diagonal term contributes `0` by construction (`qnum[i,i] = 0` from
/// [`tsne_qnum`] zeroes the product — `p[i,i]` is also `0` by the host P
/// construction, and `eps · 0 = 0` keeps the clamped-Q diagonal inert).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn tsne_grad<F: Float + CubeElement>(
    p: &Array<F>,
    qnum: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    d: u32,
    inv_qsum: F,
    eps: F,
    c_f: F,
) {
    let i = ABSOLUTE_POS_X;
    let c = ABSOLUTE_POS_Y;
    if i < n {
        if c < d {
            let yic = y[(i * d + c) as usize];
            let mut acc = F::from_int(0i64);
            let mut j = 0u32;
            while j < n {
                let idx = (i * n + j) as usize;
                let qn = qnum[idx];
                // Q = max(qnum·inv_qsum, eps) — STATEMENT-form clamp (sklearn
                // `np.maximum(Q, MACHINE_EPSILON)`).
                let mut qv = qn * inv_qsum;
                if qv < eps {
                    qv = eps;
                }
                // (P − Q)·qnum — the diagonal is inert (qnum[i,i] = 0).
                let mut pqd = (p[idx] - qv) * qn;
                if j == i {
                    pqd = F::from_int(0i64);
                }
                acc += pqd * (yic - y[(j * d + c) as usize]);
                j += 1u32;
            }
            out[(i * d + c) as usize] = c_f * acc;
        }
    }
}
