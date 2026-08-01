//! `sgd_host` — the **host** arm of the minibatch-SGD solve (MBSGD-PERF-CPU).
//!
//! ## Why a host arm exists
//! `cubecl-cpu` maps ONE OS THREAD PER UNIT (see `capability::cpu_launch_units`
//! and the KNN / HDBSCAN / UMAP / ARIMA / HGB cpu campaigns), so a GPU-shaped
//! launch grid is pathological on the cpu backend. [`sgd_solve`] is the worst
//! *shape* in the crate for that mapping — worse even than the boosting fit —
//! because its natural (and only sklearn-equivalent) minibatch size is **one
//! sample**:
//!
//! ```text
//! launches = max_iter · ceil(n / batch) · (2 + fit_intercept + 2·(tol>0) + l1)
//! ```
//!
//! At the probe's `1 000 × 16`, `batch=1`, 5 epochs that is 5 000 batches ×
//! ~5 launches = 25 000 launches for 320 kFLOP of real arithmetic. Measured
//! before this module: **80.9 s**, against scikit-learn's 0.0094 s — a factor
//! of ~8 600, essentially all of it thread-spawn. Each launch does ~64 FLOP
//! and costs ~3.2 ms.
//!
//! Worse, the shape cannot be fixed on the device side. SGD is a SEQUENTIAL
//! recurrence: sample `i+1`'s margin reads the weights sample `i` wrote, so
//! the batches cannot be merged into one wide launch without changing the
//! algorithm (which is exactly what `batch_size > 1` does, and why it is
//! documented as NOT sklearn-equivalent — see [`SgdParams::batch_size`]). The
//! only way to make the cpu backend fast here is to stop launching.
//!
//! This module replays the same recurrence in native host code: one scalar
//! loop, no threads, no barriers. There is deliberately NO worker pool — the
//! per-sample work is two passes over `d` (a dot and a fused update), i.e.
//! tens of nanoseconds, orders of magnitude below any synchronization
//! primitive.
//!
//! ## Why the result is the kernel's result (bit-identical, not merely close)
//! SGD is sequential, so a last-ULP difference compounds across `n · max_iter`
//! steps and can move the fitted iterate macroscopically. The host arm is
//! therefore an exact replay of `mlrs_kernels::sgd`, not an independent
//! implementation. Every float is produced by the same operations, in the same
//! association, in the same order:
//!
//! - the margin is the same forward `Σ_j x[·]·w[j]` seeded at zero, plus
//!   `bias` LAST ([`sgd_margin`]);
//! - the subgradient runs the same `dloss` table in `F` (not the f64 host
//!   [`dloss`] reference) with the same `F::new(1e12_f32)` clip — an **f32**
//!   literal, so at f64 the cap is `999999995904.0`, not `1e12`
//!   ([`HostFloat::lit`] reproduces that widening) ([`sgd_grad`]);
//! - the weight step keeps the kernel's association
//!   `(w[j] − ((eta·inv_b)·grad))·l2_factor`, with `eta·inv_b` hoisted out of
//!   the coordinate loop — the same product the kernel forms per coordinate,
//!   so hoisting is exact ([`sgd_weight_update`]);
//! - the cumulative-L1 shrink derives `u = u_start + (s+1)·du` from the sample
//!   counter rather than accumulating it, matching the kernel's deliberately
//!   non-loop-carried form ([`sgd_l1_shrink`]);
//! - the intercept folds `Σ_i g[i]` forward from zero ([`sgd_bias_update`]);
//! - the convergence stats fold the same strict-`>` running maxima over the
//!   same start-of-batch snapshot ([`sgd_copy`] + [`sgd_delta_max`]).
//!
//! The one structural change is that the snapshot/delta pair is FUSED into the
//! update loop when no L1 shrink runs: with `w_old` live in a register the
//! delta needs no `w_snap` array at all. That is the same value — the delta is
//! still measured against the pristine start-of-batch weight (WR-02) — it just
//! skips a `d`-element copy per batch. With L1 active the weights change again
//! after the update, so the explicit snapshot is kept.
//!
//! [`sgd_solve`]: super::sgd::sgd_solve
//! [`SgdParams::batch_size`]: super::sgd::SgdParams::batch_size
//! [`dloss`]: super::sgd::dloss
//! [`sgd_margin`]: mlrs_kernels::sgd::sgd_margin
//! [`sgd_grad`]: mlrs_kernels::sgd::sgd_grad
//! [`sgd_weight_update`]: mlrs_kernels::sgd::sgd_weight_update
//! [`sgd_l1_shrink`]: mlrs_kernels::sgd::sgd_l1_shrink
//! [`sgd_bias_update`]: mlrs_kernels::sgd::sgd_bias_update
//! [`sgd_copy`]: mlrs_kernels::sgd::sgd_copy
//! [`sgd_delta_max`]: mlrs_kernels::sgd::sgd_delta_max
//!
//! Tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use super::hgb_host::HostFloat;
use super::sgd::{loss_id, optimal_t0, schedule_eta, SgdParams, SGD_DEFAULT_MAX_ITER};

/// Whether the SGD solve should run on the host arm.
///
/// True on the cpu backend unless `MLRS_SGD_HOST=0` forces the device path
/// (the A/B knob the equivalence test drives both arms through — read via
/// [`abflag`](crate::abflag) so a test override stays thread-local).
pub(crate) fn host_solve_applicable() -> bool {
    crate::capability::active_backend_name() == "cpu"
        && crate::abflag::var("MLRS_SGD_HOST")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// Host replay of [`sgd_solve`](super::sgd::sgd_solve)'s epoch loop.
///
/// `x_host` is the `n × d` row-major design and `y_host` the length-`n` target,
/// both already materialized on the host. Returns the fitted `(coef, intercept)`
/// exactly as the device arm would leave them.
pub(crate) fn sgd_solve_host<F>(
    x_host: &[F],
    y_host: &[F],
    n: usize,
    d: usize,
    params: &SgdParams,
) -> (Vec<F>, F)
where
    F: Float + CubeElement + Pod,
{
    if size_of::<F>() == 4 {
        let (w, b) = solve_typed::<f32>(
            bytemuck::cast_slice(x_host),
            bytemuck::cast_slice(y_host),
            n,
            d,
            params,
        );
        (
            bytemuck::cast_slice::<f32, F>(&w).to_vec(),
            bytemuck::cast_slice::<f32, F>(&[b])[0],
        )
    } else {
        let (w, b) = solve_typed::<f64>(
            bytemuck::cast_slice(x_host),
            bytemuck::cast_slice(y_host),
            n,
            d,
            params,
        );
        (
            bytemuck::cast_slice::<f64, F>(&w).to_vec(),
            bytemuck::cast_slice::<f64, F>(&[b])[0],
        )
    }
}

/// Dispatch the loss ONCE, outside the sample loop, into a monomorphized solve.
///
/// The device kernel selects its `dloss` branch from a by-value `loss_id`
/// scalar on EVERY sample; on the host that same runtime switch would sit in
/// the innermost loop. Lifting it to a const generic makes the subgradient a
/// straight-line expression per instantiation (and lets the dead branches —
/// including the `exp` call — vanish), which is the host twin of specializing
/// the shader.
fn solve_typed<T: HostFloat>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    params: &SgdParams,
) -> (Vec<T>, T) {
    match loss_id(params.loss) {
        0 => solve_loss::<T, 0>(x, y, n, d, params),
        1 => solve_loss::<T, 1>(x, y, n, d, params),
        2 => solve_loss::<T, 2>(x, y, n, d, params),
        3 => solve_loss::<T, 3>(x, y, n, d, params),
        4 => solve_loss::<T, 4>(x, y, n, d, params),
        _ => solve_loss::<T, 5>(x, y, n, d, params),
    }
}

/// The `sgd_grad` kernel's `dloss` table, in `F`, for one loss family.
///
/// Mirrors `mlrs_kernels::sgd::sgd_grad` statement for statement (including
/// the `0 − y` / `(0 − 2)·y·z` forms, which are exact sign flips and so agree
/// with a plain negation, and the `F::new(1e12_f32)` clip). The f64 host
/// [`dloss`](super::sgd::dloss) is NOT reused: it computes in f64 and would
/// diverge from the device arm on the f32 path.
#[inline(always)]
fn dloss_t<T: HostFloat, const LID: u32>(p: T, y: T, epsilon: T) -> T {
    let zero = T::ZERO;
    let one = T::ONE;
    let two = T::lit(2.0);
    let mut gi = zero;
    if LID == 0 {
        // Hinge.
        let z = p * y;
        if z <= one {
            gi = zero - y;
        }
    } else if LID == 1 {
        // Log.
        gi = (zero - y) / (one + (y * p).exp());
    } else if LID == 2 {
        // Squared hinge.
        let z = one - p * y;
        if z > zero {
            gi = (zero - two) * y * z;
        }
    } else if LID == 3 {
        // Squared error.
        gi = p - y;
    } else if LID == 4 {
        // Epsilon-insensitive.
        if y - p > epsilon {
            gi = zero - one;
        }
        if p - y > epsilon {
            gi = one;
        }
    } else {
        // Squared epsilon-insensitive.
        let z = y - p;
        if z > epsilon {
            gi = (zero - two) * (z - epsilon);
        }
        if zero - z > epsilon {
            gi = two * ((zero - z) - epsilon);
        }
    }
    // The kernel's ±1e12 clip — an f32 literal in both precisions.
    let cap = T::lit(1e12);
    if gi > cap {
        gi = cap;
    }
    if gi < zero - cap {
        gi = zero - cap;
    }
    gi
}

/// The epoch/batch loop for one `(float, loss)` instantiation.
fn solve_loss<T: HostFloat, const LID: u32>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    params: &SgdParams,
) -> (Vec<T>, T) {
    let max_iter = if params.max_iter == 0 {
        SGD_DEFAULT_MAX_ITER
    } else {
        params.max_iter
    };
    let batch = params.batch_size.clamp(1, n);
    let t0 = optimal_t0(params.loss, params.alpha);
    let eps = T::from_f64(params.epsilon);

    let mut w = vec![T::ZERO; d];
    let mut bias = T::ZERO;

    let track = params.tol > 0.0;
    let l1_active = params.apply_l1 && params.l1_ratio > 0.0 && params.alpha > 0.0;
    // `w_snap` is only needed when the update loop cannot fold the delta in
    // from the live `w_old` — i.e. when an L1 shrink moves the weights again
    // afterwards, or when a batch gathers more than one sample.
    let mut w_snap = if track && (l1_active || batch > 1) {
        vec![T::ZERO; d]
    } else {
        Vec::new()
    };
    let mut q = if l1_active { vec![T::ZERO; d] } else { Vec::new() };
    let mut u_l1 = 0.0f64;

    let mut g = vec![T::ZERO; batch];

    // `t` counts SAMPLES consumed across epochs (sklearn's schedule clock).
    let mut t: u64 = 1;

    for _epoch in 0..max_iter {
        // Running epoch maxima (max |Δw|, max |w|) — the device `stats` pair,
        // zeroed at epoch start and consulted once at epoch end.
        let mut max_change = T::ZERO;
        let mut w_max = T::ZERO;

        let mut start = 0usize;
        while start < n {
            let bsz = batch.min(n - start);
            let binv = 1.0 / bsz as f64;

            // --- Pass 1 + subgradient: margin p_i over the batch rows, then
            //     g[i] = clamp(dloss(p_i, y_i)). Fused into one row scan (the
            //     two kernels are elementwise over the same `i`, so folding
            //     them changes no value). ---
            for i in 0..bsz {
                let row = &x[(start + i) * d..(start + i) * d + d];
                let mut acc = T::ZERO;
                for j in 0..d {
                    acc = acc + row[j] * w[j];
                }
                g[i] = dloss_t::<T, LID>(acc + bias, y[start + i], eps);
            }

            // --- Schedule eta for this batch (host f64; batch-start clock). ---
            let eta = schedule_eta(
                params.schedule,
                t,
                params.eta0,
                params.alpha,
                params.power_t,
                t0,
            );
            // CR-01: the per-sample L2 shrink compounded over the batch.
            let l2_factor = if params.alpha > 0.0 {
                (1.0 - (1.0 - params.l1_ratio) * eta * params.alpha)
                    .max(0.0)
                    .powi(bsz as i32)
            } else {
                1.0
            };
            // `eta · inv_b` is the same product the kernel forms per
            // coordinate, so hoisting it out of the `j` loop is exact.
            let step = T::from_f64(eta) * T::from_f64(binv);
            let l2 = T::from_f64(l2_factor);

            // The delta folds into the update loop only on the single-sample,
            // no-L1 path; every other shape needs the explicit WR-02 snapshot.
            let fuse_delta = track && bsz == 1 && !l1_active;
            if track && !fuse_delta {
                w_snap.copy_from_slice(&w);
            }

            // --- Pass 2: w[j] = (w[j] − step·Σ_i g[i]·x[i,j]) · l2_factor.
            //     The `bsz == 1` case (the only sklearn-equivalent one) is
            //     peeled so the gradient gather collapses to a single scaled
            //     row read, and — when no L1 follows — the convergence delta
            //     folds in from the live `w_old`. ---
            if bsz == 1 {
                let row = &x[start * d..start * d + d];
                let g0 = g[0];
                if fuse_delta {
                    for j in 0..d {
                        let old = w[j];
                        let grad = T::ZERO + g0 * row[j];
                        let new = (old - step * grad) * l2;
                        w[j] = new;
                        let c = (new - old).abs();
                        if c > max_change {
                            max_change = c;
                        }
                        let a = new.abs();
                        if a > w_max {
                            w_max = a;
                        }
                    }
                } else {
                    for j in 0..d {
                        let grad = T::ZERO + g0 * row[j];
                        w[j] = (w[j] - step * grad) * l2;
                    }
                }
            } else {
                for j in 0..d {
                    let mut grad = T::ZERO;
                    for i in 0..bsz {
                        grad = grad + g[i] * x[(start + i) * d + j];
                    }
                    w[j] = (w[j] - step * grad) * l2;
                }
            }

            // --- Cumulative-L1 soft-shrink (sklearn `l1penalty`). ---
            if l1_active {
                let du = params.l1_ratio * eta * params.alpha;
                let u_start = T::from_f64(u_l1);
                let du_t = T::from_f64(du);
                let zero = T::ZERO;
                for j in 0..d {
                    let mut wj = w[j];
                    let mut qj = q[j];
                    for s in 0..bsz {
                        // Derived, not loop-carried — the kernel's form.
                        let u = u_start + T::from_f64((s + 1) as f64) * du_t;
                        let z = wj;
                        if z > zero {
                            let mut cand = z - (u + qj);
                            if cand < zero {
                                cand = zero;
                            }
                            wj = cand;
                        }
                        if z < zero {
                            let mut cand = z + (u - qj);
                            if cand > zero {
                                cand = zero;
                            }
                            wj = cand;
                        }
                        qj = qj + (wj - z);
                    }
                    w[j] = wj;
                    q[j] = qj;
                }
                u_l1 += (bsz as f64) * du;
            }

            // --- Intercept step: bias -= eta·inv_b·Σ_i g_i. ---
            if params.fit_intercept {
                let mut s = T::ZERO;
                for i in 0..bsz {
                    s = s + g[i];
                }
                bias = bias - step * s;
            }

            // --- Convergence bookkeeping for the paths the update loop did
            //     not already fold (batched, or L1-shrunk weights). ---
            if track && !fuse_delta {
                for j in 0..d {
                    let c = (w[j] - w_snap[j]).abs();
                    if c > max_change {
                        max_change = c;
                    }
                    let a = w[j].abs();
                    if a > w_max {
                        w_max = a;
                    }
                }
            }

            t += bsz as u64;
            start += bsz;
        }

        if track {
            let scale = w_max.to_f64().max(1.0);
            if max_change.to_f64() <= params.tol * scale {
                break;
            }
        }
    }

    (w, bias)
}
