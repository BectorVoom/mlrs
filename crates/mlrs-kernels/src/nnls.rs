//! `nnls` — bound-constrained (non-negative) ridge solve by projected cyclic
//! coordinate descent on the GRAM, as a single-cube `#[cube(launch)]` routine
//! with the whole sweep loop and the convergence test IN KERNEL.
//!
//! This is the device arm of `Ridge(positive=True)` — sklearn's `solver='lbfgs'`
//! (`crates/mlrs-algos/src/linear/ridge_solvers.rs::nonnegative_cd` is the host
//! twin, and stays the arm for the cpu backend and for over-cap `d`).
//!
//! ## Why this can be ONE launch (unlike the Lasso/ElasticNet CD)
//! [`crate::coordinate`]'s CD kernels operate on the `n × d` DESIGN matrix, so
//! the host must drive the cyclic loop and pay a launch per coordinate. The
//! non-negative ridge objective `½‖Xw − y‖² + ½α‖w‖²` reaches `X` only through
//! the Gram, and `Ridge`'s caller has ALREADY formed `G = XᵀX` (`d×d`) and
//! `c = Xᵀy` (`d`) on-device via `prims::gram::gram_xty`. Every remaining
//! operand is `O(d²)`, so the entire solve — all `max_iter` sweeps, all `d`
//! coordinates, and the stopping test — fits inside a single cube with NO host
//! round-trip (the [`crate::jacobi_eig`] "in-kernel convergence loop" precedent,
//! D-11 gate 3). The previous path read the `d² + d` Gram back to the host,
//! solved there, and re-uploaded `coef`; this one leaves the data where
//! `gram_xty` produced it.
//!
//! ## The residual-gradient formulation (why the inner dot disappears)
//! The host twin recomputes `∂f/∂w_j = (G·w)_j − c_j + α·w_j` from scratch for
//! every coordinate — a length-`d` dot, which on a cube of `d` units would need
//! a log-tree reduction (and its barriers) per coordinate. Instead this kernel
//! carries the gradient vector `g = G·w − c` explicitly:
//!
//! ```text
//! g_j is read directly (O(1), no reduction)
//! w_j ← max(0, w_j − (g_j + α·w_j) / (G_jj + α))
//! g   ← g + Δw_j · G[j, :]        (a length-d axpy, ONE element per unit)
//! ```
//!
//! `G` is symmetric, so column `j` is row `j`: unit `i` reads `gram[j*d + i]`,
//! which is CONTIGUOUS across the cube. That leaves exactly TWO barriers per
//! coordinate (one to broadcast `Δw_j`, one so the next coordinate's `g_j` read
//! sees the completed axpy) and no reduction at all.
//!
//! ## Drift control: `g` is REBUILT at the top of every sweep
//! An incrementally-updated `g` accumulates rounding over the sweeps, which at
//! `f32` on a Gram (whose condition number is `X`'s SQUARED) is exactly where a
//! 1e-5 oracle bound gets lost. Each sweep therefore starts by recomputing
//! `g_i = Σ_k G[i,k]·w_k − c_i` from the current `w` (one row per unit, `O(d)`)
//! — bounding drift to a single sweep's worth of axpys, and making the first
//! coordinate of each sweep see BIT-for-bit the host twin's `dot(row, w) − c_j`.
//! It costs one extra `O(d)` pass against the `d` axpys already in the sweep,
//! i.e. ~2x the sweep, and it also serves as the `w = 0 ⇒ g = −c` initialisation.
//!
//! ## Iterate order matches the host twin
//! Coordinates are visited in the same ascending cyclic order with the same
//! closed-form projected update and the same `max|Δw| ≤ tol·max(1, max|w|)`
//! stop, so the two arms walk the same iterate sequence (up to the summation
//! order of the gradient rebuild). Both converge to the objective's UNIQUE
//! constrained minimiser — strictly convex for `α > 0` over a box — so the gate
//! is agreement with sklearn, not bit-identity with either arm.
//!
//! ## cubecl notes
//! - `SharedMemory::<F>::new(N)` needs a COMPILE-TIME size: `w_sh` / `g_sh` are
//!   sized to [`NNLS_MAX_DIM`] and the active region is bounded by the runtime
//!   `d` (the [`crate::jacobi_eig`] sizing + guard idiom). The host rejects
//!   `d > NNLS_MAX_DIM` before launch and takes the host twin instead.
//! - `continue` is NOT supported in `#[cube]`: the host twin's "skip a
//!   non-positive Hessian" is an `if`-wrapped update leaving `Δw_j = 0`, which
//!   makes the axpy a no-op — the same coordinate skip, branch-free.
//! - The per-sweep `max|Δw|` / `max|w|` need no reduction: unit 0 performs every
//!   coordinate's scalar update, so it accumulates both in REGISTERS across the
//!   whole sweep.
//!
//! Generic over `<F: Float + CubeElement>` and carries NO backend feature
//! (D-13). Tests live in `crates/mlrs-backend/tests/nnls_test.rs` (AGENTS.md
//! §2 — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

/// Comptime cap on the ridge feature count `d` the single-cube kernel can
/// stage. Shared memory is `2 · NNLS_MAX_DIM · size_of::<F>()` REGARDLESS of the
/// runtime `d` (a comptime allocation cannot shrink), i.e. 2 KiB at `f32` and
/// 4 KiB at `f64` — far inside every adapter's budget, unlike
/// [`crate::jacobi_eig`]'s `MAX_DIM²` tiles. The cap is instead the CUBE DIM:
/// the kernel launches one unit per feature, and 256 is the portable
/// maximum workgroup size X (wgpu's downlevel default; CUDA allows 1024).
///
/// `prims::nnls` rejects `d > NNLS_MAX_DIM` and falls back to the host twin, so
/// the cap costs supported shapes nothing.
pub const NNLS_MAX_DIM: u32 = 256;

/// Projected cyclic coordinate descent for `min_{w ≥ 0} ½·wᵀGw − cᵀw + ½α‖w‖²`,
/// the non-negative ridge normal equations, entirely in-kernel.
///
/// - `gram` is the row-major `d × d` raw Gram `G = XᵀX`. Symmetry is TRUSTED
///   (D-06) — the caller feeds `prims::gram::gram_xty`'s symmetric-by-
///   construction output, and the axpy reads row `j` AS column `j`.
/// - `xty` is the length-`d` `c = Xᵀy`.
/// - `w_out` is the length-`d` non-negative solution.
/// - `d` is the runtime feature count (`d ≤ NNLS_MAX_DIM`).
/// - `alpha` is the L2 penalty, added to the Gram DIAGONAL only (never to the
///   intercept — that is recovered post-solve by the caller, D-05).
/// - `tol` / `max_iter` are the sweep stop: `max|Δw| ≤ tol·max(1, max|w|)`, or
///   `max_iter` sweeps, whichever comes first.
///
/// Launch with ONE cube of `d` units (`CubeDim { x: d, .. }`).
#[cube(launch)]
pub fn ridge_nnls_cd<F: Float + CubeElement>(
    gram: &Array<F>,
    xty: &Array<F>,
    w_out: &mut Array<F>,
    d: u32,
    alpha: F,
    tol: F,
    max_iter: u32,
) {
    // `w` and the gradient `g = G·w − c` both live in LDS for the whole solve;
    // `bcast` carries the two cross-unit scalars (the coordinate step, and the
    // convergence flag) — no atomics, no global scratch.
    let mut w_sh = SharedMemory::<F>::new(NNLS_MAX_DIM as usize);
    let mut g_sh = SharedMemory::<F>::new(NNLS_MAX_DIM as usize);
    let mut bcast = SharedMemory::<F>::new(2usize);

    let i = UNIT_POS_X;
    let zero = F::from_int(0i64);
    let one = F::from_int(1i64);

    // w = 0. The sweep-top gradient rebuild then yields g = −c for free, so
    // there is no separate `g` initialisation to keep in sync with it.
    if i < d {
        w_sh[i as usize] = zero;
    }
    sync_cube();

    let mut sweep = 0u32;
    let mut converged = false;
    while sweep < max_iter && !converged {
        // --- Rebuild g = G·w − c from the CURRENT w (one row per unit). This
        //     is the drift control described in the module docs: every sweep
        //     restarts from an exactly-evaluated gradient rather than one
        //     carried across all previous sweeps' axpys. ---
        if i < d {
            let mut acc = zero;
            let mut k = 0u32;
            while k < d {
                acc += gram[(i * d + k) as usize] * w_sh[k as usize];
                k += 1u32;
            }
            g_sh[i as usize] = acc - xty[i as usize];
        }
        sync_cube();

        // Per-sweep maxima for the stopping test. Unit 0 performs EVERY
        // coordinate's scalar update, so these stay in its registers across the
        // whole j-loop — no shared accumulator, no tree reduction.
        let mut max_change = zero;
        let mut max_weight = zero;

        let mut j = 0u32;
        while j < d {
            // --- The scalar coordinate step, on unit 0 only. ---
            if i == 0u32 {
                let hess = gram[(j * d + j) as usize] + alpha;
                let mut delta = zero;
                // A zero feature column with α = 0 leaves this coordinate
                // unconstrained by the data; 0 is a valid feasible minimiser,
                // so the update is skipped (`continue` is unsupported in
                // `#[cube]` — the if-wrap leaves Δw = 0, a no-op axpy).
                if hess > zero {
                    let wj = w_sh[j as usize];
                    let grad = g_sh[j as usize] + alpha * wj;
                    // Projection onto w_j ≥ 0.
                    let mut next = wj - grad / hess;
                    if next < zero {
                        next = zero;
                    }
                    delta = next - wj;
                    w_sh[j as usize] = next;

                    let mut adelta = delta;
                    if adelta < zero {
                        adelta = -adelta;
                    }
                    if adelta > max_change {
                        max_change = adelta;
                    }
                    // `next` is already projected non-negative, so |next| = next.
                    if next > max_weight {
                        max_weight = next;
                    }
                }
                bcast[0usize] = delta;
            }
            // Broadcast Δw_j to the cube.
            sync_cube();

            // --- g += Δw_j · G[:, j]. G is symmetric, so column j is row j and
            //     unit i's read `gram[j*d + i]` is contiguous across the cube.
            //     Applied unconditionally: Δw_j = 0 (the skipped-coordinate
            //     case) makes it an exact no-op and costs no divergence. ---
            let delta = bcast[0usize];
            if i < d {
                g_sh[i as usize] += delta * gram[(j * d + i) as usize];
            }
            // The next coordinate reads g_sh[j+1]; it must see this axpy.
            sync_cube();

            j += 1u32;
        }

        // --- Stop test `max|Δw| ≤ tol·max(1, max|w|)` — the host twin's, and
        //     the same shape as the SAG arm's. Unit 0 owns both maxima, so it
        //     decides and publishes a flag (no `bool` in shared memory). ---
        if i == 0u32 {
            let mut scale = max_weight;
            if scale < one {
                scale = one;
            }
            let mut flag = zero;
            if max_change <= tol * scale {
                flag = one;
            }
            bcast[1usize] = flag;
        }
        sync_cube();
        if bcast[1usize] != zero {
            converged = true;
        }
        // Guard the flag against the next sweep's write before every unit has
        // read it.
        sync_cube();

        sweep += 1u32;
    }

    if i < d {
        w_out[i as usize] = w_sh[i as usize];
    }
}

/// `intercept = ȳ − x̄·coef`, computed ON-DEVICE so the `positive` fit needs no
/// host round-trip at all.
///
/// ## Why one unit, and why a serial ascending loop
/// This is `d ≤ NNLS_MAX_DIM` multiply-adds — 256 at the widest shape the
/// device solve accepts — so there is nothing to parallelize that would beat
/// the barriers a tree reduction would cost. Running it on a single unit in
/// ASCENDING `c` also reproduces the host twin's summation order exactly, which
/// is the only thing keeping the two arms comparable: the remaining difference
/// between them is purely the accumulator width (`F` here against the host's
/// `f64`), not a re-association.
///
/// That width difference is real and is why the caller keeps both arms: the
/// intercept is oracle-checked against scikit-learn at a strict `1e-5`, and an
/// `f32` dot over 256 terms whose partial sums cancel can drift further than an
/// `f64` one. `mlrs_algos::linear::ridge` gates this behind
/// `MLRS_RIDGE_HOST_INTERCEPT` for exactly that reason.
///
/// `out` is a length-1 buffer. `fit_intercept = false` is handled by the caller
/// (it never launches this).
#[cube(launch)]
pub fn ridge_intercept<F: Float + CubeElement>(
    xmean: &Array<F>,
    ymean: &Array<F>,
    coef: &Array<F>,
    out: &mut Array<F>,
    d: u32,
) {
    if UNIT_POS == 0 {
        let mut dot = F::new(0.0_f32);
        let mut c = 0u32;
        while c < d {
            dot += xmean[c as usize] * coef[c as usize];
            c += 1u32;
        }
        out[0] = ymean[0] - dot;
    }
}
