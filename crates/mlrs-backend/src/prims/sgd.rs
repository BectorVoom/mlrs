//! `sgd` — minibatch-SGD solver primitive (PRIM-10).
//!
//! Host-side orchestration of the two-pass `sgd_margin` / `sgd_weight_update`
//! kernels into the converged (pinned-epoch) SGD solve that
//! `MBSGDClassifier`/`MBSGDRegressor` consume. Mirrors the
//! [`cd_solve`](crate::prims::coordinate_descent::cd_solve) shape:
//! validate-before-launch → host epoch loop → per-batch launch → scalar/loss
//! readback → `NotConverged` at the `max_iter` cap.
//!
//! ## Flat-scalar contract (NOT the algos `SgdConfig`)
//! `mlrs-backend` does NOT depend on `mlrs-algos` (the dependency runs the other
//! way), so this prim CANNOT take the `mlrs_algos::linear::sgd_config::SgdConfig`
//! type — that would be a circular dependency. Following the `cd_solve`
//! precedent, the estimator LOWERS its `SgdConfig` into the flat scalar / enum
//! arguments here at the call site (the [`SgdLoss`] / [`SgdSchedule`] prim-local
//! enums encode the loss / schedule families the kernels need). The host
//! `dloss` table + `optimal`/`invscaling` schedule arithmetic live in this layer
//! (f64), feeding `g[]` and `eta` to the kernels.
//!
//! ## Status (Wave-1 / plan 10-02 — FILLED)
//! The host signature + a REAL geometry guard (`x.len() == n*d`, `y.len() == n`,
//! non-empty) front the epoch loop. [`sgd_solve`] drives the two-pass GATHER
//! kernels per minibatch: `sgd_margin` (pass 1, read `p[]`) → host `g[i] =
//! dloss(p_i, y_i)` → `eta = schedule_eta(t)` → host lazy-L2 / cumulative-L1
//! penalty shrink → `sgd_weight_update` (pass 2) → host intercept step;
//! `NotConverged` is surfaced at the `max_iter` cap. The kernels are
//! SharedMemory-free by construction (cubecl-cpu MLIR-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/sgd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::sgd::{sgd_margin, sgd_weight_update};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Default SGD epoch cap (sklearn `SGDClassifier`/`SGDRegressor` `max_iter`).
pub const SGD_DEFAULT_MAX_ITER: usize = 1000;

/// Default SGD stopping tolerance (sklearn `SGDClassifier`/`SGDRegressor`
/// `tol`). The pinned oracle OVERRIDES this with `tol = 0` + a fixed `max_iter`
/// so neither side early-stops (Pitfall 2/7).
pub const SGD_DEFAULT_TOL: f64 = 1e-3;

/// The loss family the host `dloss` table selects on (the estimator lowers its
/// typed `mlrs_algos` `Loss` into this prim-local enum at the call site — the
/// prim layer cannot depend on algos). Wave-1 wires the per-sample gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdLoss {
    /// Hinge `max(0, 1 - y·p)`.
    Hinge,
    /// Log `log(1 + exp(-y·p))`.
    Log,
    /// Squared hinge `max(0, 1 - y·p)²`.
    SquaredHinge,
    /// Squared error `½(p - y)²`.
    SquaredError,
    /// Epsilon-insensitive `max(0, |y - p| - ε)`.
    EpsilonInsensitive,
    /// Squared epsilon-insensitive `max(0, |y - p| - ε)²`.
    SquaredEpsilonInsensitive,
}

/// The learning-rate schedule the host step arithmetic selects on (lowered from
/// the algos `LearningRate` at the call site).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdSchedule {
    /// Bottou `optimal` `η(t) = 1/(α·(t₀ + t − 1))`.
    Optimal,
    /// Inverse-scaling `η(t) = eta0 / t^power_t`.
    InvScaling,
    /// Constant `η = eta0`.
    Constant,
    /// Adaptive `η = eta0`, halved on a stalled objective.
    Adaptive,
}

/// Flat-scalar penalty / schedule arguments the estimator lowers its `SgdConfig`
/// into (the prim cannot take the algos `SgdConfig` — circular dependency). All
/// scalars are host f64; the host `dloss`/schedule math runs in f64 and feeds the
/// kernels.
#[derive(Debug, Clone, Copy)]
pub struct SgdParams {
    /// The loss family (prim-local enum, lowered from algos `Loss`).
    pub loss: SgdLoss,
    /// The learning-rate schedule (prim-local enum, lowered from algos
    /// `LearningRate`).
    pub schedule: SgdSchedule,
    /// L2 penalty strength.
    pub alpha: f64,
    /// ElasticNet L1/L2 mixing `∈ [0, 1]`.
    pub l1_ratio: f64,
    /// Whether the host applies the L1 shrink (`l1_ratio > 0` / penalty includes
    /// L1).
    pub apply_l1: bool,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
    /// Initial learning rate `eta0`.
    pub eta0: f64,
    /// Inverse-scaling exponent `power_t`.
    pub power_t: f64,
    /// Epsilon-insensitive margin.
    pub epsilon: f64,
    /// Minibatch size.
    pub batch_size: usize,
    /// Epoch cap.
    pub max_iter: usize,
    /// Stopping tolerance (`0` ⇒ run all `max_iter` epochs).
    pub tol: f64,
}

/// Solve the minibatch-SGD problem on the design `x` (`n × d`, row-major) and
/// target `y` (length `n`), returning the device-resident pair
/// `(coef, intercept)` (`coef` length `d`, `intercept` length 1).
///
/// - Geometry (`n*d == x.len()`, `y.len() == n`, non-empty) is validated BEFORE
///   any unsafe launch (ASVS V5 / T-10-01-02); a wrong shape returns
///   [`PrimError::ShapeMismatch`] so a malformed shape can never reach a device
///   launch even at scaffold stage.
/// - `params` carries the lowered (flat-scalar) penalty / schedule the host
///   gradient + step math reads.
///
/// ## Compute (Wave-1)
/// Epoch loop over `max_iter`; per minibatch (natural row order, `shuffle=false`):
/// `sgd_margin::launch` → read `p[]` → host `g[i] = dloss(p_i, y_i)` clipped ±1e12
/// → `eta = schedule_eta(t)` → host lazy-L2 / cumulative-L1 penalty shrink →
/// `sgd_weight_update::launch` → host intercept `b -= eta·(Σ g_i)·inv_b`. Tracks
/// the max coefficient change; stops when `< tol·scale`, else runs to the cap and
/// the iterate is returned as-is (sklearn's `tol=0` deterministic-epochs contract;
/// the caller maps a non-convergence to its estimator-level error if it cares).
#[allow(clippy::too_many_arguments)]
pub fn sgd_solve<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    params: &SgdParams,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = shape;

    // --- Geometry guard (ASVS V5 / T-10-01-02): validate BEFORE any launch so a
    //     malformed shape is a recoverable typed error, not an out-of-bounds
    //     device read. Mirrors cd_solve / laplacian. ---
    validate_geometry(x.len(), y.len(), n, d)?;

    let elem = size_of::<F>();
    let client = pool.client().clone();

    // Host targets (read ONCE; the per-sample dloss is host f64 — the margin/
    // gradient device kernels carry the n/d-heavy work, the host owns the loss).
    let y_host: Vec<f64> = y.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    // Host design (read ONCE): the batch kernels index from row 0 of the array
    // they are launched over, so each minibatch's contiguous row block is uploaded
    // into a reused batch buffer (the cubecl 0.10 `ArrayArg` has no offset variant;
    // copying the slice keeps the kernel single-purpose — row 0-based, no
    // row-offset scalar). The raw `F` values are kept so the upload is byte-exact.
    let x_raw: Vec<F> = x.to_host(pool);

    let max_iter = if params.max_iter == 0 {
        SGD_DEFAULT_MAX_ITER
    } else {
        params.max_iter
    };
    let batch = params.batch_size.clamp(1, n);
    let inv_b = 1.0 / batch as f64;

    // The Bottou t0 (optimal schedule only); computed once before the loop.
    let t0 = optimal_t0(params.loss, params.alpha);

    // Device-resident solver state: w (len d), reused every batch; a length-`batch`
    // margin output + a length-`batch·d` batch-design buffer, both reused.
    let mut w_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &vec![narrow::<F>(0.0); d]);
    let p_handle = pool.acquire(batch * elem); // margin output, reused.

    // Host coefficient mirror (for the convergence delta + the L1 cumulative
    // soft-shrink u-accumulator state); the device w is the authoritative copy the
    // kernel updates, read back per batch to drive the host penalty shrink.
    let mut bias = 0.0f64;
    let mut u_l1 = 0.0f64; // cumulative L1 penalty budget (sklearn wscale trick).
    let mut q = vec![0.0f64; d]; // applied-L1 per coordinate (sklearn `q`).

    // `t` counts SAMPLES consumed across epochs (sklearn's schedule clock).
    let mut t: u64 = 1;

    let cube_block = 256u32;

    for _epoch in 0..max_iter {
        let mut max_change = 0.0f64;
        let mut w_max = 0.0f64;

        let mut start = 0usize;
        while start < n {
            let bsz = batch.min(n - start);
            let binv = 1.0 / bsz as f64;

            // Upload this batch's contiguous row block [start, start+bsz) into a
            // device batch buffer (the kernels index from row 0 of the array they
            // receive, so the batch slice must start at offset 0; cubecl 0.10's
            // `ArrayArg` has no offset variant). The upload is a bounded same-size
            // allocation per batch, served from the free-list after warmup and
            // released back at batch end — the memory gate asserts this conserves.
            let slice = &x_raw[start * d..(start + bsz) * d];
            let xb = DeviceArray::<ActiveRuntime, F>::from_host(pool, slice);

            // --- Pass 1: margin p[i] = Σ_j x[i,j]·w[j] + bias over the batch rows. ---
            let p_arg = unsafe { ArrayArg::from_raw_parts(p_handle.clone(), bsz) };
            let x_off = unsafe { ArrayArg::from_raw_parts(xb.handle().clone(), bsz * d) };
            let w_arg = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
            let count = CubeCount::Static(
                ((bsz as u32) + cube_block - 1) / cube_block.max(1),
                1,
                1,
            );
            let dim = CubeDim {
                x: cube_block,
                y: 1,
                z: 1,
            };
            sgd_margin::launch::<F, ActiveRuntime>(
                &client,
                count.clone(),
                dim,
                x_off,
                w_arg,
                narrow::<F>(bias),
                p_arg,
                bsz as u32,
                d as u32,
            );
            let p_dev = DeviceArray::<ActiveRuntime, F>::from_raw(p_handle.clone(), bsz);
            let p_host: Vec<f64> = p_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();

            // --- Host per-sample gradient g[i] = dloss(p_i, y_i), clipped ±1e12. ---
            let g: Vec<F> = (0..bsz)
                .map(|i| {
                    let gi = dloss(params.loss, p_host[i], y_host[start + i], params.epsilon)
                        .clamp(-1e12, 1e12);
                    narrow::<F>(gi)
                })
                .collect();

            // --- Schedule eta for this batch (use the batch-start sample clock). ---
            let eta = schedule_eta(
                params.schedule,
                t,
                params.eta0,
                params.alpha,
                params.power_t,
                t0,
            );

            // --- WR-02: snapshot the TRUE start-of-batch weights (BEFORE any
            //     gradient step or penalty shrink this batch). The convergence
            //     delta is diffed against this pristine snapshot so `max_change`
            //     reflects the FULL per-batch update (gradient + penalty) and the
            //     `tol`-based early stop does not fire prematurely. ---
            let w_start: Vec<f64> =
                w_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();

            // --- Pass 2: w[j] -= eta·inv_b·Σ_i g[i]·x[i,j]. ---
            let g_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &g);
            let x_off2 = unsafe { ArrayArg::from_raw_parts(xb.handle().clone(), bsz * d) };
            let g_arg = unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), bsz) };
            let w_arg2 = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
            let count2 = CubeCount::Static(
                ((d as u32) + cube_block - 1) / cube_block.max(1),
                1,
                1,
            );
            sgd_weight_update::launch::<F, ActiveRuntime>(
                &client,
                count2,
                dim,
                x_off2,
                g_arg,
                w_arg2,
                narrow::<F>(eta),
                narrow::<F>(binv),
                d as u32,
                bsz as u32,
            );
            g_dev.release_into(pool);

            // --- Host lazy-L2 `wscale` shrink applied AFTER the gradient step:
            //     w_j *= max(0, 1 - (1 - l1_ratio)·eta·alpha). Order matches
            //     sklearn `_plain_sgd` / RESEARCH §Per-sample update sequence
            //     (the penalty shrink follows the gradient step, before the L1
            //     cumulative soft-shrink). Applied to the RESULTING w via a host
            //     round-trip (small d). ---
            let l2_factor = (1.0 - (1.0 - params.l1_ratio) * eta * params.alpha).max(0.0);
            if params.alpha > 0.0 && l2_factor != 1.0 {
                let mut w_host: Vec<f64> =
                    w_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                for wj in w_host.iter_mut() {
                    *wj *= l2_factor;
                }
                let w_shrunk: Vec<F> = w_host.iter().map(|&v| narrow::<F>(v)).collect();
                let w_re = DeviceArray::<ActiveRuntime, F>::from_host(pool, &w_shrunk);
                // Swap the device w to the freshly-shrunk buffer (release the old).
                let old_w = std::mem::replace(&mut w_dev, w_re);
                old_w.release_into(pool);
            }

            // --- Host cumulative-L1 soft-shrink (sklearn `l1penalty`): u grows by
            //     l1_ratio·eta·alpha; each w_j is pulled toward 0 by the budget. ---
            if params.apply_l1 && params.l1_ratio > 0.0 && params.alpha > 0.0 {
                u_l1 += params.l1_ratio * eta * params.alpha;
                let mut w_after: Vec<f64> =
                    w_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                for j in 0..d {
                    let z = w_after[j];
                    if w_after[j] > 0.0 {
                        w_after[j] = (w_after[j] - (u_l1 + q[j])).max(0.0);
                    } else if w_after[j] < 0.0 {
                        w_after[j] = (w_after[j] + (u_l1 - q[j])).min(0.0);
                    }
                    q[j] += w_after[j] - z;
                }
                let w_l1: Vec<F> = w_after.iter().map(|&v| narrow::<F>(v)).collect();
                let w_re2 = DeviceArray::<ActiveRuntime, F>::from_host(pool, &w_l1);
                let old = std::mem::replace(&mut w_dev, w_re2);
                old.release_into(pool);
            }

            // --- Host intercept step: b -= eta·inv_b·Σ_i g_i (intercept_decay = 1.0
            //     dense, A3). ---
            if params.fit_intercept {
                let g_sum: f64 = (0..bsz).map(|i| host_to_f64(g[i])).sum();
                bias -= eta * binv * g_sum;
            }

            // --- Convergence bookkeeping: max |Δw| this batch, measured against
            //     the pristine start-of-batch snapshot (WR-02) so the delta
            //     reflects the FULL update (gradient + L2 + L1), not a
            //     penalty-mutated intermediate. ---
            let w_new: Vec<f64> =
                w_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
            for j in 0..d {
                let change = (w_new[j] - w_start[j]).abs();
                if change > max_change {
                    max_change = change;
                }
                let aw = w_new[j].abs();
                if aw > w_max {
                    w_max = aw;
                }
            }

            // Release the batch design buffer back to the pool (bounded same-size
            // alloc reused next batch — the memory gate's conservation signal).
            xb.release_into(pool);

            t += bsz as u64;
            let _ = inv_b;
            start += bsz;
        }

        // sklearn's cheap host stopping gate: max coefficient change vs tol·scale.
        if params.tol > 0.0 {
            let scale = w_max.max(1.0);
            if max_change <= params.tol * scale {
                break;
            }
        }
    }

    pool.release(p_handle, batch * elem);

    // Materialize the device-resident intercept scalar.
    let intercept: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &[narrow::<F>(bias)]);

    Ok((w_dev, intercept))
}

/// Per-sample loss subgradient `dloss(p, y)` (RESEARCH §SGD-Math, the exact
/// sklearn `_sgd_fast` table). Computed host-side per minibatch sample, fed to
/// `sgd_weight_update` as `g[]`. `epsilon` is the epsilon-insensitive margin
/// (ignored by the other losses).
pub fn dloss(loss: SgdLoss, p: f64, y: f64, epsilon: f64) -> f64 {
    match loss {
        SgdLoss::Hinge => {
            let z = p * y;
            if z <= 1.0 {
                -y
            } else {
                0.0
            }
        }
        SgdLoss::SquaredHinge => {
            let z = 1.0 - p * y;
            if z > 0.0 {
                -2.0 * y * z
            } else {
                0.0
            }
        }
        SgdLoss::Log => -y / (1.0 + (y * p).exp()),
        SgdLoss::SquaredError => p - y,
        SgdLoss::EpsilonInsensitive => {
            if y - p > epsilon {
                -1.0
            } else if p - y > epsilon {
                1.0
            } else {
                0.0
            }
        }
        SgdLoss::SquaredEpsilonInsensitive => {
            let z = y - p;
            if z > epsilon {
                -2.0 * (z - epsilon)
            } else if z < -epsilon {
                2.0 * (-z - epsilon)
            } else {
                0.0
            }
        }
    }
}

/// The Bottou `optimal` schedule `t0` (`BaseSGD._init_t` / `_sgd_fast`
/// `optimal_init`, RESEARCH lines 510-514, A1):
/// `typw = sqrt(1/sqrt(alpha)); initial_eta0 = typw / max(1, |dloss(loss,-typw,1)|);
/// t0 = 1/(initial_eta0·alpha)`. For `alpha <= 0` (the convex-objective case with
/// alpha≈0) this is unused (the schedule is constant/invscaling), so it returns a
/// harmless `1.0` rather than dividing by zero.
pub fn optimal_t0(loss: SgdLoss, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 1.0;
    }
    let typw = (1.0 / alpha.sqrt()).sqrt();
    let initial_eta0 = typw / dloss(loss, -typw, 1.0, 0.1).abs().max(1.0);
    1.0 / (initial_eta0 * alpha)
}

/// The learning-rate schedule `eta(t)` (RESEARCH §SGD-Math). `t` is the 1-based
/// sample clock. `Adaptive` is treated as `Constant` here (the no-improvement
/// halving is driven by the host loop, which currently runs the deterministic
/// pinned schedule; the estimator may extend it).
pub fn schedule_eta(
    lr: SgdSchedule,
    t: u64,
    eta0: f64,
    alpha: f64,
    power_t: f64,
    t0: f64,
) -> f64 {
    match lr {
        SgdSchedule::Optimal => 1.0 / (alpha * (t0 + t as f64 - 1.0)),
        SgdSchedule::InvScaling => eta0 / (t as f64).powf(power_t),
        SgdSchedule::Constant => eta0,
        SgdSchedule::Adaptive => eta0,
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side scalar math.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("sgd is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn narrow<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("sgd is f32/f64 only"),
    }
}

/// Validate the SGD solve geometry (ASVS V5 / T-10-01-02): `x` must be a
/// well-formed `n × d` matrix (`n*d == x.len()`), `y` must be length `n`, and
/// neither dimension may be zero. A malformed shape is rejected at the boundary
/// (no well-defined solve) BEFORE any device launch.
fn validate_geometry(x_len: usize, y_len: usize, n: usize, d: usize) -> Result<(), PrimError> {
    if n == 0 || d == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if y_len != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: y_len,
        });
    }
    Ok(())
}
