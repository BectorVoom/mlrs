//! `lbfgs` — stable symmetric-multinomial softmax loss + gradient kernel
//! (LINEAR-05, D-12). The genuinely-new device piece of the L-BFGS solver: a
//! feature-free `#[cube]` kernel that emits ONLY the multinomial logistic loss +
//! gradient. The two-loop recursion + strong-Wolfe line search live HOST-side in
//! `prims::lbfgs` (D-10); the kernel never iterates the solver.
//!
//! ## Objective (pinned, `_linear_loss.py:48-64,226-229` + `_loss/loss.py`)
//! For the row-major design `x` (`n × d`), the K full weight vectors `w` (`K × d`,
//! the SYMMETRIC over-parameterized form — D-12, so binary is the K=2 case), the
//! per-class biases `b` (length `K`), the integer labels `y` (length `n`) and the
//! L2 strength `l2_reg = 1/(C·n)`:
//!
//! ```text
//! raw[i,k] = x[i]·w[k] + b[k]
//! m[i]     = max_k raw[i,k]                              (Pitfall 4 — stabilize)
//! lse[i]   = m[i] + log Σ_k exp(raw[i,k] − m[i])
//! loss     = (1/n) Σ_i (lse[i] − raw[i,y_i]) + ½·l2_reg·‖w‖_F²   (b UNPENALIZED — Pitfall 3)
//! p[i,k]   = exp(raw[i,k] − lse[i])                     (softmax)
//! gradW[k] = (1/n) Σ_i (p[i,k] − [y_i==k])·x[i]  + l2_reg·w[k]
//! gradb[k] = (1/n) Σ_i (p[i,k] − [y_i==k])           (NO penalty)
//! ```
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! The cpu(f64) backend's MLIR lowering rejects `SharedMemory` + mutable `bool`
//! flags + `F::INFINITY` consts + descending-shift loops + cross-unit atomics
//! (plans 05-02/05-05 hit this). This kernel runs entirely on unit 0 of a single
//! cube with ONLY `F`/`u32` accumulators and `if`-guarded forward loops — no
//! `SharedMemory`, no `bool`, no infinity sentinel, no atomics/scatter. The row
//! max for the log-sum-exp is an `if`-guarded forward scan seeded at `raw[i,0]`
//! (correct: a real logit is a valid seed). The `n`/`K`/`d` of the LogReg fits in
//! Phase 5 are modest, so the single-unit GATHER is acceptable and keeps the
//! reduction order identical to numpy's left-to-right accumulation.
//!
//! Generic over `<F: Float + CubeElement>`, carrying NO backend feature (D-13).
//! The Wave-0 scaffold owns the `pub mod lbfgs;` line in `lib.rs`; this file adds
//! its own `pub use` (file-disjoint, parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/lbfgs_test.rs` (AGENTS.md §2 — never
//! an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::softmax_loss_grad as lbfgs_softmax_loss_grad;

/// Stable symmetric-multinomial softmax loss + gradient (D-12).
///
/// Given the row-major `n × d` design `x`, the `K × d` weight matrix `w`, the
/// length-`K` bias `b`, the length-`n` integer labels `y` (stored as `F`, e.g.
/// `2.0` for class 2), and `l2_reg = 1/(C·n)`, writes:
///
/// - `loss_out[0]` = `(1/n) Σ_i (lse[i] − raw[i,y_i]) + ½·l2_reg·‖w‖_F²`
///   (intercept UNPENALIZED — Pitfall 3).
/// - `grad_w` (`K × d`, row-major) = `(1/n)(P − Y)ᵀX + l2_reg·W`.
/// - `grad_b` (`K`) = `(1/n) Σ_i (p[i] − Y_onehot[i])` (NO penalty).
///
/// - `n`, `d`, `k_classes` are scalar args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg`, mirroring `coordinate.rs`'s `rows: u32`).
/// - `l2_reg` is the scalar `F` strength passed by value.
///
/// Launched as a SINGLE cube; only unit 0 acts (single-cube GATHER — no atomics,
/// no `SharedMemory`). The log-sum-exp subtracts the row max BEFORE `exp`
/// (Pitfall 4) so well-separated classes never overflow.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn softmax_loss_grad<F: Float + CubeElement>(
    x: &Array<F>,
    w: &Array<F>,
    b: &Array<F>,
    y: &Array<F>,
    loss_out: &mut Array<F>,
    grad_w: &mut Array<F>,
    grad_b: &mut Array<F>,
    n: u32,
    d: u32,
    k_classes: u32,
    l2_reg: F,
) {
    // Only unit 0 acts (single-cube GATHER — cubecl-cpu MLIR-safe).
    if UNIT_POS == 0 {
        let zero = F::from_int(0i64);
        let one = F::from_int(1i64);
        let half = F::new(0.5_f32);
        let inv_n = one / F::cast_from(n);

        // --- Zero the gradient accumulators (grad_w is K×d, grad_b is K). ---
        let mut gk = 0u32;
        while gk < k_classes {
            grad_b[gk as usize] = zero;
            let mut gj = 0u32;
            while gj < d {
                grad_w[(gk * d + gj) as usize] = zero;
                gj += 1u32;
            }
            gk += 1u32;
        }

        let mut loss_acc = zero;

        // --- Per-row pass: stable log-sum-exp, softmax, loss + gradient GATHER. ---
        let mut i = 0u32;
        while i < n {
            // raw[i,k] = x[i]·w[k] + b[k]; track the row max (forward if-scan).
            let mut row_max = zero;
            let mut k = 0u32;
            while k < k_classes {
                let mut raw_ik = b[k as usize];
                let mut j = 0u32;
                while j < d {
                    raw_ik += x[(i * d + j) as usize] * w[(k * d + j) as usize];
                    j += 1u32;
                }
                // Seed the max with class 0's logit, then keep the larger.
                if k == 0u32 {
                    row_max = raw_ik;
                } else if raw_ik > row_max {
                    row_max = raw_ik;
                }
                k += 1u32;
            }

            // sum_exp = Σ_k exp(raw[i,k] − row_max) (stable; ≥ 1).
            let mut sum_exp = zero;
            let mut k2 = 0u32;
            while k2 < k_classes {
                let mut raw_ik = b[k2 as usize];
                let mut j = 0u32;
                while j < d {
                    raw_ik += x[(i * d + j) as usize] * w[(k2 * d + j) as usize];
                    j += 1u32;
                }
                sum_exp += (raw_ik - row_max).exp();
                k2 += 1u32;
            }
            // lse[i] = row_max + log(sum_exp) (natural log).
            let lse_i = row_max + sum_exp.ln();

            // y_i as a u32 class index (labels stored as whole-number F).
            let yi = u32::cast_from(y[i as usize]);

            // loss += lse[i] − raw[i, y_i].
            let mut raw_iy = b[yi as usize];
            let mut jy = 0u32;
            while jy < d {
                raw_iy += x[(i * d + jy) as usize] * w[(yi * d + jy) as usize];
                jy += 1u32;
            }
            loss_acc += lse_i - raw_iy;

            // Gradient GATHER: p[i,k] = exp(raw[i,k] − lse[i]);
            //   diff = p[i,k] − [y_i==k]; grad_b[k] += diff; grad_w[k,j] += diff·x[i,j].
            let mut k3 = 0u32;
            while k3 < k_classes {
                let mut raw_ik = b[k3 as usize];
                let mut j = 0u32;
                while j < d {
                    raw_ik += x[(i * d + j) as usize] * w[(k3 * d + j) as usize];
                    j += 1u32;
                }
                let p_ik = (raw_ik - lse_i).exp();
                // [y_i == k] indicator without a bool flag.
                let mut diff = p_ik;
                if k3 == yi {
                    diff = p_ik - one;
                }
                grad_b[k3 as usize] += diff;
                let mut jj = 0u32;
                while jj < d {
                    grad_w[(k3 * d + jj) as usize] += diff * x[(i * d + jj) as usize];
                    jj += 1u32;
                }
                k3 += 1u32;
            }

            i += 1u32;
        }

        // --- Scale by 1/n, add the L2 penalty to grad_w (NOT grad_b), and the
        //     ½·l2_reg·‖w‖_F² term to the loss (intercept unpenalized). ---
        let mut w_norm2 = zero;
        let mut sk = 0u32;
        while sk < k_classes {
            grad_b[sk as usize] = grad_b[sk as usize] * inv_n;
            let mut sj = 0u32;
            while sj < d {
                let idx = (sk * d + sj) as usize;
                let wkj = w[idx];
                w_norm2 += wkj * wkj;
                grad_w[idx] = grad_w[idx] * inv_n + l2_reg * wkj;
                sj += 1u32;
            }
            sk += 1u32;
        }

        loss_out[0] = loss_acc * inv_n + half * l2_reg * w_norm2;
    }
}
