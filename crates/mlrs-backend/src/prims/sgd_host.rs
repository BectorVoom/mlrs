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
//! - the loss-plateau `tol`/`n_iter_no_change` stop (MBSGD-PERF-WGPU) sums the
//!   same per-sample loss VALUE table ([`sgd_loss`]) forward from zero within
//!   a batch ([`sgd_sumloss`]), then carries `best_loss`/`no_improvement_count`
//!   across epochs exactly as sklearn's `_plain_sgd` does — this REPLACED an
//!   earlier coefficient-delta check (`sgd_copy` + `sgd_delta_max`, since
//!   removed) that, unlike sklearn, could take hundreds of extra epochs to
//!   satisfy on data where the training loss had already plateaued.
//!
//! Every summation keeps the kernel's order; [`wide_dot`] is the one
//! deliberate, opt-in exception (a reassociation, off by default).
//!
//! [`sgd_solve`]: super::sgd::sgd_solve
//! [`SgdParams::batch_size`]: super::sgd::SgdParams::batch_size
//! [`dloss`]: super::sgd::dloss
//! [`sgd_margin`]: mlrs_kernels::sgd::sgd_margin
//! [`sgd_grad`]: mlrs_kernels::sgd::sgd_grad
//! [`sgd_weight_update`]: mlrs_kernels::sgd::sgd_weight_update
//! [`sgd_l1_shrink`]: mlrs_kernels::sgd::sgd_l1_shrink
//! [`sgd_bias_update`]: mlrs_kernels::sgd::sgd_bias_update
//! [`sgd_loss`]: mlrs_kernels::sgd::sgd_loss
//! [`sgd_sumloss`]: mlrs_kernels::sgd::sgd_sumloss
//!
//! Tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use super::hgb_host::HostFloat;
use super::sgd::{loss_id, optimal_t0, schedule_eta, SgdParams, SGD_DEFAULT_MAX_ITER};

/// Whether the SGD solve should run on the host arm.
///
/// True on the cpu AND wgpu backends unless `MLRS_SGD_HOST=0` forces the
/// device path (the A/B knob the equivalence test drives both arms through —
/// read via [`abflag`](crate::abflag) so a test override stays thread-local).
///
/// ## Why wgpu joined cpu (MBSGD-PERF-WGPU)
/// The per-sample launch count documented in this module's header
/// (`max_iter · ceil(n/batch) · (2 + fit_intercept + 2·(tol>0) + l1)`) is not a
/// cpu-thread-per-unit artifact — it is `sgd_solve`'s device-kernel shape,
/// full stop. On the actual wgpu target (measured, not extrapolated from cpu —
/// see the verify-on-target-hardware rule) a tiny `n=200, d=8, max_iter=5` fit
/// (5 000 launches) took multiple SECONDS wall against ~0.27 s of CPU time —
/// i.e. the process was blocked on GPU submission/sync, not computing. wgpu
/// has no cheaper way to issue a launch than cpu does here, and `batch_size ==
/// 1` (the only sklearn-equivalent setting) cannot be widened without changing
/// the algorithm (see the module docs), so the cpu fix — stop launching,
/// replay the SAME recurrence on the host — applies verbatim. Only `x`/`y`
/// (downloaded once) and the fitted `w`/`bias` (uploaded once) cross the
/// device boundary; cuda/rocm are left on the device path (untestable in this
/// environment — do not extrapolate this decision onto them without measuring
/// on that hardware).
pub(crate) fn host_solve_applicable() -> bool {
    matches!(crate::capability::active_backend_name(), "cpu" | "wgpu")
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

/// The `sgd_loss` kernel's loss-VALUE table, in `F` (as opposed to
/// [`dloss_t`]'s derivative) — mirrors `mlrs_kernels::sgd::sgd_loss` statement
/// for statement, including the `Log` branch's "compute the unclipped default,
/// then overwrite in the two extreme regions" structure (a transient overflow
/// to `+inf` in the discarded default is harmless IEEE arithmetic on either
/// arm). The f64 host [`loss_value`](super::sgd::loss_value) is NOT reused,
/// same reasoning as [`dloss_t`].
#[inline(always)]
fn loss_value_t<T: HostFloat, const LID: u32>(p: T, y: T, epsilon: T) -> T {
    let zero = T::ZERO;
    let one = T::ONE;
    let mut li = zero;
    if LID == 0 {
        // Hinge: max(0, 1 - p*y).
        let z = one - p * y;
        if z > zero {
            li = z;
        }
    } else if LID == 1 {
        // Log: log(1 + exp(-p*y)), clipped at |z| = 18 like sklearn.
        let z = p * y;
        let eighteen = T::lit(18.0);
        li = (zero - z).exp().log1p();
        if z > eighteen {
            li = (zero - z).exp();
        }
        if z < zero - eighteen {
            li = zero - z;
        }
    } else if LID == 2 {
        // Squared hinge: max(0, 1 - p*y)^2.
        let z = one - p * y;
        if z > zero {
            li = z * z;
        }
    } else if LID == 3 {
        // Squared error: 0.5*(p-y)^2.
        let d = p - y;
        li = T::lit(0.5) * d * d;
    } else if LID == 4 {
        // Epsilon-insensitive: max(0, |y-p|-eps).
        let z = (y - p).abs() - epsilon;
        if z > zero {
            li = z;
        }
    } else {
        // Squared epsilon-insensitive: max(0, |y-p|-eps)^2.
        let z = (y - p).abs() - epsilon;
        if z > zero {
            li = z * z;
        }
    }
    li
}

/// Lane count of the [`wide_dot`] accumulator (one AVX2 f32 register).
const DOT_LANES: usize = 8;

/// `Σ_j x[j]·w[j]` over `DOT_LANES` INDEPENDENT accumulators, reduced in lane
/// order, with the `d % DOT_LANES` tail folded in ascending afterwards.
///
/// ## Why this is not the default
/// The serial forward sum in [`solve_loss`] is a dependency chain: each add
/// waits on the previous one (3-4 cycles), so a `d = 64` margin costs ~192
/// cycles of pure latency and IS essentially the whole per-sample time — the
/// coordinate update beside it vectorizes and takes ~8. Splitting the chain
/// into 8 lanes cuts that to ~24 cycles plus the reduction, which measured
/// ~2.4× on the fit at `d = 64`.
///
/// It is opt-in (`MLRS_SGD_WIDE_DOT=1`) because it is a REASSOCIATION: floating
/// point addition is not associative, so this returns a (slightly) different
/// margin from the `sgd_margin` kernel, and SGD's sequential recurrence
/// compounds that difference across `n · max_iter` steps. Turning it on
/// therefore breaks the bit-identity that `sgd_host_equivalence_test` pins, and
/// makes a cpu fit differ from the same fit on wgpu/CUDA/ROCm.
///
/// How much it actually moves the fit depends entirely on HOW the loss reads
/// the margin, which splits the table in two (measured, `d = 64`, hinge and
/// squared-hinge and log, both float types, vs the serial default):
///
/// | loss | margin enters via | max relative Δcoef |
/// |---|---|---|
/// | hinge (the default), ε-insensitive | a THRESHOLD (`p·y ≤ 1`) | **0** — bit-identical |
/// | log | the value (`−y/(1+exp(y·p))`) | 1e-4 … 8e-4 |
/// | squared hinge, squared error, squared ε-insensitive | the value | 1.5e-2 … 7e-2 |
///
/// The threshold losses discard the margin's exact value — the subgradient is
/// `−y` or `0` either way — so a last-ULP change cannot propagate at all,
/// except in the vanishing case where `p·y` lands within one ULP of the
/// comparison boundary and flips the branch. That residual case is precisely
/// why hinge does not get this by default either: a probabilistically
/// bit-identical arm is a flaky contract, not a contract.
///
/// The direction of the error is favourable — sklearn's own
/// `WeightVector.dot` accumulates into a `double` even on the float32 path, so
/// a lane-split f32 sum sits CLOSER to the oracle than the serial one — but
/// "closer to sklearn, different from our own device arm" is a trade a caller
/// should make deliberately, not one the default should make for them.
#[inline(always)]
fn wide_dot<T: HostFloat>(row: &[T], w: &[T], d: usize) -> T {
    let mut acc = [T::ZERO; DOT_LANES];
    let chunks = d / DOT_LANES;
    for c in 0..chunks {
        let base = c * DOT_LANES;
        for (l, a) in acc.iter_mut().enumerate() {
            *a = *a + row[base + l] * w[base + l];
        }
    }
    let mut s = acc[0];
    for a in acc.iter().skip(1) {
        s = s + *a;
    }
    for j in chunks * DOT_LANES..d {
        s = s + row[j] * w[j];
    }
    s
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
    // Opt-in reassociated margin — see [`wide_dot`] for why it is not default.
    let wide_dot_on = crate::abflag::is_on("MLRS_SGD_WIDE_DOT");

    let mut w = vec![T::ZERO; d];
    let mut bias = T::ZERO;

    let track = params.tol > 0.0;
    let l1_active = params.apply_l1 && params.l1_ratio > 0.0 && params.alpha > 0.0;
    let mut q = if l1_active {
        vec![T::ZERO; d]
    } else {
        Vec::new()
    };
    let mut u_l1 = 0.0f64;

    let mut g = vec![T::ZERO; batch];

    // `t` counts SAMPLES consumed across epochs (sklearn's schedule clock).
    let mut t: u64 = 1;
    // sklearn's `best_loss`/`no_improvement_count` (`tol > 0` only) — carried
    // ACROSS epochs, unlike `sumloss` below.
    let mut best_loss = f64::INFINITY;
    let mut no_improvement_count: usize = 0;
    let n_iter_no_change = params.n_iter_no_change.max(1);

    for _epoch in 0..max_iter {
        // Running epoch loss accumulator — the device `sumloss[0]`, zeroed at
        // epoch start and consulted once at epoch end.
        let mut sumloss = T::ZERO;

        let mut start = 0usize;
        while start < n {
            let bsz = batch.min(n - start);
            let binv = 1.0 / bsz as f64;

            // --- Pass 1 + subgradient: margin p_i over the batch rows, then
            //     g[i] = clamp(dloss(p_i, y_i)). Fused into one row scan (the
            //     two kernels are elementwise over the same `i`, so folding
            //     them changes no value). When tracking, the loss VALUE
            //     (`sgd_loss`) is folded in too — same `p_i`/`y_i`, a SEPARATE
            //     batch-local sum (`batch_sumloss`) added into `sumloss` AFTER
            //     the batch, mirroring the device's two-kernel/two-launch
            //     structure (`sgd_loss` then `sgd_sumloss`) exactly. ---
            let mut batch_sumloss = T::ZERO;
            for i in 0..bsz {
                let row = &x[(start + i) * d..(start + i) * d + d];
                let acc = if wide_dot_on {
                    wide_dot(row, &w, d)
                } else {
                    let mut acc = T::ZERO;
                    for j in 0..d {
                        acc = acc + row[j] * w[j];
                    }
                    acc
                };
                let margin = acc + bias;
                g[i] = dloss_t::<T, LID>(margin, y[start + i], eps);
                if track {
                    batch_sumloss = batch_sumloss + loss_value_t::<T, LID>(margin, y[start + i], eps);
                }
            }
            if track {
                sumloss = sumloss + batch_sumloss;
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

            // --- Pass 2: w[j] = (w[j] − step·Σ_i g[i]·x[i,j]) · l2_factor.
            //     The `bsz == 1` case (the only sklearn-equivalent one) is
            //     peeled so the gradient gather collapses to a single scaled
            //     row read. ---
            if bsz == 1 {
                let row = &x[start * d..start * d + d];
                let g0 = g[0];
                for j in 0..d {
                    let grad = T::ZERO + g0 * row[j];
                    w[j] = (w[j] - step * grad) * l2;
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

            t += bsz as u64;
            start += bsz;
        }

        // sklearn's training-loss-plateau stop: carried best/no-improvement
        // state exactly as `_plain_sgd` does.
        if track {
            let sumloss64 = sumloss.to_f64();
            if sumloss64 > best_loss - params.tol * n as f64 {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }
            if sumloss64 < best_loss {
                best_loss = sumloss64;
            }
            if no_improvement_count >= n_iter_no_change {
                break;
            }
        }
    }

    (w, bias)
}
