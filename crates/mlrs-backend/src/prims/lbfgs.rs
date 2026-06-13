//! `prims::lbfgs` — host-driven L-BFGS solver + softmax loss/grad orchestration
//! (LINEAR-05). THE highest correctness risk in the project (RESEARCH §"L-BFGS").
//!
//! Two pieces:
//!
//! 1. [`lbfgs_minimize`] — a GENERIC host L-BFGS minimizer parameterized by a
//!    closure `f(x) -> (loss, grad)` returning the objective value + gradient at a
//!    host parameter vector. It owns the standard two-loop recursion with an
//!    `m = 10` history of `(s, y)` pairs and a strong-Wolfe (Moré-Thuente-style)
//!    line search, using the scipy L-BFGS-B constants pinned in RESEARCH:
//!    `gtol = 1e-4` on `max |proj-grad|` (here just `max |g_i|`, unbounded),
//!    `ftol = 64·eps` relative-f decrease, `maxls = 50`, `maxiter = 100`. The
//!    gradient + `(s, y)` history buffers are acquired ONCE and reused every
//!    iteration; the convergence test reads exactly ONE scalar (the max-abs
//!    gradient) per iteration (D-10). On `maxiter` reached it returns the last
//!    iterate with `converged = false` (the estimator surfaces
//!    [`AlgoError::NotConverged`] if needed — plan 05-10).
//!
//! 2. [`softmax_loss_grad`] — the host launcher for the Task-1
//!    `mlrs_kernels::lbfgs` stable-softmax kernel: validates the geometry of the
//!    `(x, y, w, b)` operands BEFORE any unsafe launch (ASVS V5 / T-05-06-01),
//!    launches the kernel, and reads back the loss + `(gradW, gradb)`. The LogReg
//!    estimator (plan 05-10) wraps this in a closure for [`lbfgs_minimize`].
//!
//! ## Why the convex-quadratic standalone validation FIRST (Pitfall 5)
//! `lbfgs_minimize` is validated standalone on a convex quadratic `½xᵀAx − bᵀx`
//! (gradient `Ax − b`, unique minimizer `x* = A⁻¹b`) BEFORE the softmax path —
//! the convex objective converges to a unique global minimum, so the final
//! iterate must equal `A⁻¹b` within 1e-5 regardless of small line-search
//! differences, isolating "is my L-BFGS correct" from "does it match sklearn's
//! path". See `crates/mlrs-backend/tests/lbfgs_test.rs`.
//!
//! The Wave-0 scaffold owns the `pub mod lbfgs;` line in `prims/mod.rs`; this file
//! adds its own `pub use` (file-disjoint, parallel-safe). Tests live in
//! `crates/mlrs-backend/tests/lbfgs_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::lbfgs::lbfgs_softmax_loss_grad;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// scipy L-BFGS-B `maxcor` (history size `m`) — `scipy/optimize/_lbfgsb_py.py:96`.
pub const LBFGS_M: usize = 10;
/// scipy L-BFGS-B `maxls` (max line-search evaluations).
pub const LBFGS_MAXLS: usize = 50;
/// scipy L-BFGS-B `maxiter` cap (sklearn `LogisticRegression` `max_iter`).
pub const LBFGS_MAXITER: usize = 100;
/// scipy L-BFGS-B `pgtol`/`gtol` (sklearn `tol`): max |proj-grad| stop.
pub const LBFGS_GTOL: f64 = 1e-4;
/// scipy L-BFGS-B `ftol = factr·eps = 64·eps` relative-f decrease stop.
pub const LBFGS_FTOL: f64 = 64.0 * f64::EPSILON;

/// Strong-Wolfe line-search constants (`c1` sufficient decrease, `c2` curvature)
/// — the standard L-BFGS values (Nocedal & Wright; scipy uses the same family).
const WOLFE_C1: f64 = 1e-4;
const WOLFE_C2: f64 = 0.9;

/// WR-01: the precise reason [`lbfgs_minimize`] stopped. `converged` alone cannot
/// distinguish a legitimate ftol stall (a genuine stationary point) from a
/// line-search BREAKDOWN at a non-stationary point (a NaN/degenerate gradient) —
/// both leave `converged = false` with `iters < maxiter`, so the estimator would
/// wrongly accept the breakdown as success. The estimator must surface
/// `NotConverged` on [`LbfgsStopReason::LineSearchFailed`] regardless of `iters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsStopReason {
    /// `max_grad <= gtol` was reached — a true first-order stationary point.
    Converged,
    /// scipy's `ftol` relative-f-decrease stall (a flat/near-stationary point;
    /// for the gauge-degenerate symmetric softmax this is the legitimate stop).
    FtolStall,
    /// The strong-Wolfe line search could not find an acceptable step — a
    /// breakdown at a possibly NON-stationary point. NOT a success.
    LineSearchFailed,
    /// The `maxiter` cap was reached without any of the above stops.
    MaxIter,
}

/// The outcome of an [`lbfgs_minimize`] run: the final iterate `x`, its objective
/// value `loss`, the max-abs gradient at it, the iteration count, whether the
/// `gtol` convergence criterion was met before `maxiter` (`converged`), and the
/// precise [`LbfgsStopReason`] (WR-01).
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Final parameter vector (the last iterate).
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub loss: f64,
    /// `max_i |g_i|` at `x` (the convergence scalar).
    pub max_grad: f64,
    /// Number of outer L-BFGS iterations performed.
    pub iters: usize,
    /// Whether `max_grad <= gtol` was reached before `maxiter` (else the estimator
    /// may surface [`mlrs_core`]`::AlgoError::NotConverged`).
    pub converged: bool,
    /// WR-01: the precise stop reason, so the estimator can distinguish a benign
    /// ftol stall from a line-search breakdown (which it must reject).
    pub stop_reason: LbfgsStopReason,
}

/// Minimize `f` (returning `(loss, grad)` at a host parameter vector) by L-BFGS
/// with the scipy L-BFGS-B constants (`m = 10`, `gtol = 1e-4`, `ftol = 64·eps`,
/// `maxls = 50`, `maxiter = 100`) — the standard two-loop recursion + a
/// strong-Wolfe line search (RESEARCH §"Pattern 2 (host loop)").
///
/// - `x0` is the starting iterate (length `n`, validated non-empty before use —
///   T-05-06-01 / ASVS V5).
/// - `f` is evaluated to get `(loss, grad)`; for the LogReg path it launches the
///   Task-1 softmax kernel ([`softmax_loss_grad`]) and assembles the flat gradient.
/// - The `(s, y)` history (`≤ m` pairs) and the working gradient/direction vectors
///   are owned by THIS function and reused every iteration (D-10 bounded
///   allocation; no per-iteration device array is read — the only per-iteration
///   readback is the closure's own one-scalar pattern when device-backed).
///
/// Returns the last iterate even when `maxiter` is hit (with `converged = false`),
/// so the estimator decides whether to surface `NotConverged`.
pub fn lbfgs_minimize<Fobj>(
    x0: Vec<f64>,
    mut f: Fobj,
    gtol: f64,
    ftol: f64,
    maxls: usize,
    maxiter: usize,
) -> Result<LbfgsResult, PrimError>
where
    Fobj: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    // T-05-06-01 / ASVS V5: a zero-length parameter vector is a geometry error.
    if n == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.x0",
            rows: 0,
            cols: 1,
            len: 0,
        });
    }

    let mut x = x0;
    let (mut loss, mut grad) = f(&x);
    if grad.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.grad",
            rows: n,
            cols: 1,
            len: grad.len(),
        });
    }

    // History of (s, y) pairs + their ρ = 1/(yᵀs); ring of capacity m, reused.
    let m = LBFGS_M;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    // Reused per-iteration working buffers (acquired once — D-10 bounded alloc).
    let mut q = vec![0.0f64; n];
    let mut alpha = vec![0.0f64; m];
    let mut direction = vec![0.0f64; n];

    let mut max_grad = max_abs(&grad); // ← the ONE scalar convergence test.
    if max_grad <= gtol {
        return Ok(LbfgsResult {
            x,
            loss,
            max_grad,
            iters: 0,
            converged: true,
            stop_reason: LbfgsStopReason::Converged,
        });
    }

    let maxiter = if maxiter == 0 { LBFGS_MAXITER } else { maxiter };
    let mut converged = false;
    // WR-01: assume the cap until a more specific stop fires below.
    let mut stop_reason = LbfgsStopReason::MaxIter;
    let mut iters = 0usize;

    for k in 0..maxiter {
        iters = k + 1;

        // --- Two-loop recursion: r = H_k · (−grad). q starts at the gradient. ---
        q.copy_from_slice(&grad);
        let n_hist = s_hist.len();
        // First loop (most recent → oldest).
        for i in (0..n_hist).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling γ = (sᵀy)/(yᵀy) from the most recent pair (=1
        // on the first iteration, the steepest-descent step).
        let gamma = if n_hist > 0 {
            let last = n_hist - 1;
            let sy = dot(&s_hist[last], &y_hist[last]);
            let yy = dot(&y_hist[last], &y_hist[last]);
            if yy > 0.0 {
                sy / yy
            } else {
                1.0
            }
        } else {
            1.0
        };
        for j in 0..n {
            q[j] *= gamma;
        }
        // Second loop (oldest → most recent).
        for i in 0..n_hist {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_hist[i]);
        }
        // Descent direction d = −r.
        for j in 0..n {
            direction[j] = -q[j];
        }

        // Guard: d must be a descent direction (gᵀd < 0); else reset to −grad.
        let mut g_dot_d = dot(&grad, &direction);
        if g_dot_d >= 0.0 {
            for j in 0..n {
                direction[j] = -grad[j];
            }
            g_dot_d = dot(&grad, &direction);
        }

        // --- Strong-Wolfe line search for a step `t` along `direction`. ---
        let ls = line_search_wolfe(&mut f, &x, loss, &direction, g_dot_d, maxls);
        let (t, new_loss, new_grad) = match ls {
            Some(v) => v,
            None => {
                // Line search failed to find an acceptable step — a breakdown at
                // a possibly non-stationary point. WR-01: record the distinct stop
                // reason so the estimator surfaces NotConverged regardless of how
                // many iterations ran (it is NOT a benign ftol stall).
                stop_reason = LbfgsStopReason::LineSearchFailed;
                break;
            }
        };

        // s = t·direction ; y = new_grad − grad ; x ← x + s ; grad ← new_grad.
        let mut s = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        for j in 0..n {
            s[j] = t * direction[j];
            y[j] = new_grad[j] - grad[j];
            x[j] += s[j];
        }

        let prev_loss = loss;
        loss = new_loss;
        grad = new_grad;
        max_grad = max_abs(&grad); // ← the ONE scalar convergence test.

        // Push the (s, y) pair onto the bounded history (ring of capacity m).
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            // Skip a non-curvature-positive update (keeps H positive definite).
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        // --- Convergence: gtol on max |grad|, ftol on relative-f decrease. ---
        if max_grad <= gtol {
            converged = true;
            stop_reason = LbfgsStopReason::Converged;
            break;
        }
        let f_decrease = (prev_loss - loss).abs();
        let denom = prev_loss.abs().max(loss.abs()).max(1.0);
        if f_decrease / denom <= ftol {
            // Negligible relative-f decrease — scipy's `ftol` stop.
            converged = max_grad <= gtol;
            stop_reason = if converged {
                LbfgsStopReason::Converged
            } else {
                LbfgsStopReason::FtolStall
            };
            break;
        }
    }

    Ok(LbfgsResult {
        x,
        loss,
        max_grad,
        iters,
        converged,
        stop_reason,
    })
}

/// Strong-Wolfe line search (sufficient decrease `c1` + curvature `c2`) along
/// `direction` from `x`. Returns `(t, loss_at_x+t·d, grad_at_x+t·d)` on success.
///
/// A bracketing/zoom scheme (Nocedal & Wright Alg. 3.5/3.6 simplified): grow the
/// step until the Wolfe conditions bracket a minimizer, then bisect to satisfy
/// the curvature condition, capped at `maxls` objective evaluations.
#[allow(clippy::too_many_arguments)]
fn line_search_wolfe<Fobj>(
    f: &mut Fobj,
    x: &[f64],
    loss0: f64,
    direction: &[f64],
    g_dot_d0: f64,
    maxls: usize,
) -> Option<(f64, f64, Vec<f64>)>
where
    Fobj: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x.len();
    let phi0 = loss0;
    let dphi0 = g_dot_d0; // φ'(0) = gradᵀd < 0.
    if dphi0 >= 0.0 {
        return None;
    }

    let eval = |f: &mut Fobj, t: f64| -> (f64, f64, Vec<f64>) {
        let mut xt = vec![0.0f64; n];
        for j in 0..n {
            xt[j] = x[j] + t * direction[j];
        }
        let (l, g) = f(&xt);
        let dphi = dot(&g, direction);
        (l, dphi, g)
    };

    let mut t_prev = 0.0f64;
    let mut phi_prev = phi0;
    let mut t = 1.0f64; // L-BFGS unit-step initial guess.
    let t_max = 1e10f64;
    let mut evals = 0usize;

    loop {
        if evals >= maxls {
            return None;
        }
        let (phi_t, dphi_t, grad_t) = eval(f, t);
        evals += 1;

        // Armijo (sufficient decrease) violated, or non-decreasing vs previous →
        // a minimizer is bracketed in (t_prev, t): zoom.
        if phi_t > phi0 + WOLFE_C1 * t * dphi0 || (evals > 1 && phi_t >= phi_prev) {
            return zoom(
                f, x, direction, phi0, dphi0, t_prev, phi_prev, t, maxls, evals,
            );
        }
        // Strong curvature condition satisfied → accept.
        if dphi_t.abs() <= -WOLFE_C2 * dphi0 {
            return Some((t, phi_t, grad_t));
        }
        // Overshot the minimizer (positive slope) → zoom with reversed bracket.
        if dphi_t >= 0.0 {
            return zoom(
                f, x, direction, phi0, dphi0, t, phi_t, t_prev, maxls, evals,
            );
        }

        t_prev = t;
        phi_prev = phi_t;
        t = (t * 2.0).min(t_max);
        if t >= t_max {
            // Cannot grow further; accept the last point if it decreased.
            let (phi_last, _, grad_last) = eval(f, t);
            if phi_last < phi0 {
                return Some((t, phi_last, grad_last));
            }
            return None;
        }
    }
}

/// The "zoom" stage of the strong-Wolfe search: bisect the bracket `(t_lo, t_hi)`
/// until the Wolfe conditions hold (Nocedal & Wright Alg. 3.6, bisection variant).
#[allow(clippy::too_many_arguments)]
fn zoom<Fobj>(
    f: &mut Fobj,
    x: &[f64],
    direction: &[f64],
    phi0: f64,
    dphi0: f64,
    mut t_lo: f64,
    mut phi_lo: f64,
    mut t_hi: f64,
    maxls: usize,
    mut evals: usize,
) -> Option<(f64, f64, Vec<f64>)>
where
    Fobj: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x.len();
    let eval = |f: &mut Fobj, t: f64| -> (f64, f64, Vec<f64>) {
        let mut xt = vec![0.0f64; n];
        for j in 0..n {
            xt[j] = x[j] + t * direction[j];
        }
        let (l, g) = f(&xt);
        let dphi = dot(&g, direction);
        (l, dphi, g)
    };

    while evals < maxls {
        let t = 0.5 * (t_lo + t_hi);
        let (phi_t, dphi_t, grad_t) = eval(f, t);
        evals += 1;

        if phi_t > phi0 + WOLFE_C1 * t * dphi0 || phi_t >= phi_lo {
            t_hi = t;
        } else {
            if dphi_t.abs() <= -WOLFE_C2 * dphi0 {
                return Some((t, phi_t, grad_t));
            }
            if dphi_t * (t_hi - t_lo) >= 0.0 {
                t_hi = t_lo;
            }
            t_lo = t;
            phi_lo = phi_t;
        }
        if (t_hi - t_lo).abs() < 1e-14 {
            // Bracket collapsed — accept the current point if it decreased.
            if phi_t < phi0 {
                return Some((t, phi_t, grad_t));
            }
            return None;
        }
    }
    None
}

/// Launch the Task-1 stable-softmax kernel and read back `(loss, gradW, gradb)`.
///
/// Validates the geometry of every operand BEFORE any unsafe launch (ASVS V5 /
/// T-05-06-01): `x.len() == n*d`, `y.len() == n`, `w.len() == k*d`, `b.len() == k`
/// → [`PrimError::ShapeMismatch`]. The kernel runs on a single cube (unit 0);
/// `l2_reg = 1/(C·n)` is the caller's responsibility. Returns the host scalars +
/// flat gradient vectors the LogReg estimator's closure assembles for
/// [`lbfgs_minimize`].
#[allow(clippy::too_many_arguments)]
pub fn softmax_loss_grad<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    w: &DeviceArray<ActiveRuntime, F>,
    b: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
    l2_reg: f64,
) -> Result<(f64, Vec<f64>, Vec<f64>), PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-05-06-01: validate geometry BEFORE any unsafe launch. ---
    validate_softmax_geometry(x.len(), y.len(), w.len(), b.len(), n, d, k)?;

    let elem = size_of::<F>();
    let client = pool.client().clone();

    // Outputs: loss (length 1), gradW (k*d), gradb (k) — acquired here.
    let loss_handle = pool.acquire(elem);
    let grad_w_handle = pool.acquire(k * d * elem);
    let grad_b_handle = pool.acquire(k * elem);

    let cube1 = CubeCount::Static(1, 1, 1);
    let dim1 = CubeDim { x: 1, y: 1, z: 1 };

    unsafe {
        let x_arg = ArrayArg::from_raw_parts(x.handle().clone(), n * d);
        let w_arg = ArrayArg::from_raw_parts(w.handle().clone(), k * d);
        let b_arg = ArrayArg::from_raw_parts(b.handle().clone(), k);
        let y_arg = ArrayArg::from_raw_parts(y.handle().clone(), n);
        let loss_arg = ArrayArg::from_raw_parts(loss_handle.clone(), 1);
        let gw_arg = ArrayArg::from_raw_parts(grad_w_handle.clone(), k * d);
        let gb_arg = ArrayArg::from_raw_parts(grad_b_handle.clone(), k);
        lbfgs_softmax_loss_grad::launch::<F, ActiveRuntime>(
            &client,
            cube1,
            dim1,
            x_arg,
            w_arg,
            b_arg,
            y_arg,
            loss_arg,
            gw_arg,
            gb_arg,
            n as u32,
            d as u32,
            k as u32,
            from_f64::<F>(l2_reg),
        );
    }

    // Read back the loss scalar + the two gradient vectors.
    let loss_dev = DeviceArray::<ActiveRuntime, F>::from_raw(loss_handle.clone(), 1);
    let loss = host_to_f64(loss_dev.to_host_metered(pool)[0]);

    let gw_dev = DeviceArray::<ActiveRuntime, F>::from_raw(grad_w_handle.clone(), k * d);
    let grad_w: Vec<f64> = gw_dev
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();

    let gb_dev = DeviceArray::<ActiveRuntime, F>::from_raw(grad_b_handle.clone(), k);
    let grad_b: Vec<f64> = gb_dev
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();

    pool.release(loss_handle, elem);
    pool.release(grad_w_handle, k * d * elem);
    pool.release(grad_b_handle, k * elem);

    Ok((loss, grad_w, grad_b))
}

/// Validate the softmax operand geometry (ASVS V5 / T-05-06-01): `x` is `n×d`,
/// `y` is length `n`, `w` is `k×d`, `b` is length `k` — all checked BEFORE any
/// unsafe launch.
fn validate_softmax_geometry(
    x_len: usize,
    y_len: usize,
    w_len: usize,
    b_len: usize,
    n: usize,
    d: usize,
    k: usize,
) -> Result<(), PrimError> {
    if n == 0 || d == 0 || k == 0 || n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if y_len != n {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.y",
            rows: n,
            cols: 1,
            len: y_len,
        });
    }
    if k.checked_mul(d).map(|v| v != w_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.w",
            rows: k,
            cols: d,
            len: w_len,
        });
    }
    if b_len != k {
        return Err(PrimError::ShapeMismatch {
            operand: "lbfgs.b",
            rows: k,
            cols: 1,
            len: b_len,
        });
    }
    // WR-03: n, d, k are cast to u32 for the kernel launch geometry; reject an
    // overflowing dimension BEFORE launch so the cast cannot silently truncate
    // into an out-of-bounds device read.
    for (operand, dim) in [("lbfgs.n", n), ("lbfgs.d", d), ("lbfgs.k", k)] {
        if dim > u32::MAX as usize {
            return Err(PrimError::ShapeMismatch {
                operand,
                rows: dim,
                cols: 0,
                len: u32::MAX as usize,
            });
        }
    }
    Ok(())
}

/// `max_i |v_i|` — the L-BFGS convergence scalar (max projected gradient; with no
/// bounds the "projected" gradient is just the gradient).
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
}

/// Host dot product `Σ_i a_i·b_i` (left-to-right, numpy order).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// In-place axpy `v ← v + a·u`.
fn axpy(v: &mut [f64], a: f64, u: &[f64]) {
    for i in 0..v.len() {
        v[i] += a * u[i];
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side scalar math.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("lbfgs is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn from_f64<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("lbfgs is f32/f64 only"),
    }
}
