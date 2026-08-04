//! `sgd` — minibatch-SGD solver primitive (PRIM-10).
//!
//! Host-side orchestration of the device SGD kernels into the converged
//! (pinned-epoch) SGD solve that `MBSGDClassifier`/`MBSGDRegressor` consume.
//! Mirrors the [`cd_solve`](crate::prims::coordinate_descent::cd_solve) shape:
//! validate-before-launch → host epoch loop → per-batch launch →
//! `NotConverged` at the `max_iter` cap.
//!
//! ## Flat-scalar contract (NOT the algos `SgdConfig`)
//! `mlrs-backend` does NOT depend on `mlrs-algos` (the dependency runs the other
//! way), so this prim CANNOT take the `mlrs_algos::linear::sgd_config::SgdConfig`
//! type — that would be a circular dependency. Following the `cd_solve`
//! precedent, the estimator LOWERS its `SgdConfig` into the flat scalar / enum
//! arguments here at the call site (the [`SgdLoss`] / [`SgdSchedule`] prim-local
//! enums encode the loss / schedule families the kernels need). The
//! `optimal`/`invscaling` schedule arithmetic lives in this layer (f64), feeding
//! `eta` (and the compounded L2 factor) to the kernels as by-value scalars.
//!
//! ## Device-resident epoch loop (the cuML-parity requirement)
//! The epoch loop is LAUNCH-ONLY: per minibatch it enqueues
//! `sgd_margin` → `sgd_grad` (the on-device `dloss` table) →
//! `sgd_weight_update` (gradient step with the fused lazy-L2 shrink) →
//! optional `sgd_l1_shrink` (cumulative-L1 soft-shrink with device `q[]`) →
//! optional `sgd_bias_update` (device intercept), with NO device↔host
//! synchronization inside the batch loop. The previous implementation read the
//! margin, the weights (up to 3×), and re-uploaded the batch design + penalty-
//! shrunk weights EVERY batch — on CUDA that made training latency-bound
//! (PCIe/launch round-trips), not compute-bound, which is why fits ran far
//! slower than cuML. Now:
//!
//! - `x`/`y` stay on device untouched; batches are addressed IN PLACE via a
//!   `row_offset` kernel scalar (no per-batch slice upload).
//! - The `tol > 0` convergence gate (MBSGD-PERF-WGPU) is the sklearn
//!   training-loss-plateau check, NOT a coefficient delta: `sgd_loss` computes
//!   the per-sample loss VALUE (as opposed to `sgd_grad`'s derivative) and
//!   `sgd_sumloss` folds it into a running length-1 `sumloss` epoch
//!   accumulator, read back ONCE PER EPOCH — the only steady-state readback —
//!   and compared against a host-carried `best_loss`/`no_improvement_count`
//!   exactly as sklearn's `_plain_sgd` does. With `tol == 0` (the
//!   pinned-oracle mode) the solve performs ZERO readbacks until the caller
//!   materializes the fitted state.
//! - The host `dloss`/`loss_value` tables ([`dloss`]/[`loss_value`]) are
//!   retained as the f64 references the device `sgd_grad`/`sgd_loss` kernels
//!   are tested against (and for the `optimal`-schedule `t0` probe).
//!
//! Tests live in `crates/mlrs-backend/tests/sgd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_core::{f64_to_host, host_to_f64};
use mlrs_kernels::sgd::{
    sgd_bias_update, sgd_grad, sgd_l1_shrink, sgd_loss, sgd_margin, sgd_sumloss,
    sgd_weight_update,
};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Default SGD epoch cap (sklearn `SGDClassifier`/`SGDRegressor` `max_iter`).
pub const SGD_DEFAULT_MAX_ITER: usize = 1000;

/// Default SGD stopping tolerance (sklearn `SGDClassifier`/`SGDRegressor`
/// `tol`). The pinned oracle OVERRIDES this with `tol = 0` + a fixed `max_iter`
/// so neither side early-stops (Pitfall 2/7).
pub const SGD_DEFAULT_TOL: f64 = 1e-3;

/// The loss family the `sgd_grad` device kernel selects on (the estimator
/// lowers its typed `mlrs_algos` `Loss` into this prim-local enum at the call
/// site — the prim layer cannot depend on algos).
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
/// scalars are host f64; the host schedule math runs in f64 and feeds the
/// kernels by-value scalars.
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
    /// Whether to apply the L1 shrink (`l1_ratio > 0` / penalty includes L1).
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
    ///
    /// WR-03 — NOT sklearn-equivalent for `batch_size > 1`: sklearn's
    /// `SGDClassifier`/`SGDRegressor` are strictly PER-SAMPLE (effective batch of
    /// 1) and re-read the margin between every sample. This prim, for
    /// `batch_size > 1`, takes a SINGLE averaged gradient step per minibatch
    /// (summed gradient scaled by `1/bsz`) and does NOT re-margin between samples
    /// within the batch. The L2 / L1 penalty budgets are compounded per sample
    /// (CR-01 / CR-02) so the penalty path tracks the sample count, but the
    /// averaged-gradient + no-mid-batch-re-margin model is a DIFFERENT algorithm
    /// from sklearn for `batch_size > 1`. Only `batch_size == 1` reproduces
    /// sklearn's `coef_`/`intercept_` to the oracle tolerance.
    pub batch_size: usize,
    /// Epoch cap.
    pub max_iter: usize,
    /// Stopping tolerance (`0` ⇒ run all `max_iter` epochs).
    pub tol: f64,
    /// Consecutive non-improving epochs (`tol > 0` only) before stopping —
    /// sklearn `n_iter_no_change`. Compared against a per-epoch `sumloss` (the
    /// sklearn training-loss-plateau check), not the coefficient delta.
    pub n_iter_no_change: usize,
}

/// The `sgd_grad` kernel's loss selector for a [`SgdLoss`] (the single source of
/// the id ↔ loss mapping; the kernel doc table mirrors this).
pub fn loss_id(loss: SgdLoss) -> u32 {
    match loss {
        SgdLoss::Hinge => 0,
        SgdLoss::Log => 1,
        SgdLoss::SquaredHinge => 2,
        SgdLoss::SquaredError => 3,
        SgdLoss::EpsilonInsensitive => 4,
        SgdLoss::SquaredEpsilonInsensitive => 5,
    }
}

/// Whether [`sgd_solve`] runs on the HOST arm for the active backend — i.e.
/// whether the design matrix would only be uploaded to be read straight back.
///
/// Callers holding `x` on the host already (a borrowed Arrow buffer at the
/// Python boundary, say) should branch on this and use
/// [`sgd_solve_host_slice`], which skips the round-trip entirely. On a real
/// device backend this is `false` and the caller must upload as usual.
///
/// See `prims::sgd_host` for why the cpu AND wgpu backends have a host arm at
/// all (cuda/rocm are untested and stay on the device path).
pub fn sgd_host_available() -> bool {
    super::sgd_host::host_solve_applicable()
}

/// [`sgd_solve`] over HOST slices, returning the fitted `(coef, intercept)` on
/// the host — the no-upload entry point for callers that already hold `x`/`y`
/// in host memory.
///
/// Only valid where [`sgd_host_available`] is true; it is a hard error
/// otherwise, since there is no host implementation of the device kernels to
/// fall back to.
///
/// ## Why this exists (MBSGD-PERF-CPU)
/// `DeviceArray::from_host` copies, and `DeviceArray::to_host` copies TWICE
/// (`read_one` materializes a byte buffer, then `cast_slice(..).to_vec()`
/// materializes the typed one). Routing a host-resident design through the
/// `DeviceArray` boundary just to reach the host arm therefore costs three
/// full passes over `x`. Measured on the `50 000 × 64` f32 probe those passes
/// were 10 ms against 6 ms of actual solving — the ingress, not the
/// arithmetic, was the dominant term. This entry point removes all three.
///
/// Geometry is validated exactly as in [`sgd_solve`] (ASVS V5 / T-10-01-02).
pub fn sgd_solve_host_slice<F>(
    x: &[F],
    y: &[F],
    shape: (usize, usize),
    params: &SgdParams,
) -> Result<(Vec<F>, F), PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = shape;
    // No device-transcendental guard here: this entry point is host-only by
    // construction (it never falls back to a device kernel), so it has no
    // dependency on what the active backend's shader compiler supports.
    validate_geometry(x.len(), y.len(), n, d)?;
    if !sgd_host_available() {
        return Err(PrimError::UnsupportedCapability {
            operand: "sgd_solve_host_slice",
            capability: "a host-resident SGD solve (this entry point is cpu/wgpu-only)",
        });
    }
    Ok(super::sgd_host::sgd_solve_host::<F>(x, y, n, d, params))
}

/// Solve the minibatch-SGD problem on the design `x` (`n × d`, row-major) and
/// target `y` (length `n`), returning the device-resident pair
/// `(coef, intercept)` (`coef` length `d`, `intercept` length 1).
///
/// - Geometry (`n*d == x.len()`, `y.len() == n`, non-empty) is validated BEFORE
///   any unsafe launch (ASVS V5 / T-10-01-02); a wrong shape returns
///   [`PrimError::ShapeMismatch`] so a malformed shape can never reach a device
///   launch.
/// - `params` carries the lowered (flat-scalar) penalty / schedule the host
///   step math reads.
///
/// ## Compute (device-resident)
/// Epoch loop over `max_iter`; per minibatch (natural row order,
/// `shuffle=false`): `sgd_margin` → `sgd_grad` (`g[i] = dloss(p_i, y_i)`
/// clipped ±1e12, on device) → `eta = schedule_eta(t)` (host f64) →
/// `sgd_weight_update` (gradient step × compounded lazy-L2 factor) → optional
/// `sgd_l1_shrink` (cumulative-L1, device `q[]`) → optional `sgd_bias_update`
/// → optional `sgd_loss` + `sgd_sumloss` (loss-plateau tracking). No host
/// synchronization inside the batch loop. The training-loss-plateau stopping
/// gate (`tol > 0`) is folded on device (`sgd_loss` + `sgd_sumloss`) and read
/// ONCE per epoch against a host `best_loss`/`no_improvement_count`; with
/// `tol == 0` the solve runs all `max_iter` epochs readback-free and the
/// iterate is returned as-is (sklearn's `tol=0` deterministic-epochs contract;
/// the caller maps a non-convergence to its estimator-level error if it
/// cares).
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

    // --- cpu/wgpu host arm (MBSGD-PERF-CPU/WGPU): the launch-only epoch loop
    //     below is pathological under cubecl-cpu's one-thread-per-unit mapping
    //     AND under wgpu's per-launch submission/sync cost — the
    //     sklearn-equivalent `batch_size == 1` issues ~5 launches per SAMPLE for
    //     ~2·d FLOP of work (measured on the actual wgpu adapter: a trivial
    //     `n=200, d=8, max_iter=5` fit took several SECONDS wall against ~0.27 s
    //     of cpu time — blocked on GPU submission, not computing). `sgd_host`
    //     replays the SAME recurrence, value for value, in a scalar host loop.
    //     See `sgd_host`'s module docs for the equivalence argument (bit-exact
    //     on cpu; a tight tolerance on wgpu, see `sgd_host_equivalence_test`)
    //     and the measured factor.
    //
    //     This check runs BEFORE the device f64-transcendental guard below on
    //     purpose: that guard exists because compiling `F::exp` into a GPU
    //     shader can fail on an adapter without 64-bit transcendentals, but the
    //     host arm never touches a device shader — it calls the host libm
    //     `exp`/`ln`/`tanh` regardless of what the active backend's shader
    //     compiler supports. Gating the host arm behind that guard would
    //     needlessly block f64 log-loss (etc.) MBSGD fits on exactly this kind
    //     of adapter even though the host arm has no dependency on the
    //     limitation at all. ---
    if super::sgd_host::host_solve_applicable() {
        let x_host = x.to_host(pool);
        let y_host = y.to_host(pool);
        let (w, b) = super::sgd_host::sgd_solve_host::<F>(&x_host, &y_host, n, d, params);
        return Ok((
            DeviceArray::from_host(pool, &w),
            DeviceArray::from_host(pool, &[b]),
        ));
    }

    // --- Capability guard, ALL LOSSES (device arm only, from here on): the
    //     log-loss gradient `-y / (1 + exp(y·p))` (`mlrs_kernels::sgd`) is the
    //     only branch that evaluates a transcendental, but the loss is a
    //     RUNTIME switch inside ONE kernel — so `F::exp` is compiled into the
    //     shader whichever loss is selected, and a backend without f64 `exp`
    //     fails at SHADER-COMPILE time, not at execution. Gating on
    //     `params.loss == Log` therefore does NOT help: a hinge-loss f64 fit
    //     crashes just the same (measured — it is what
    //     `mbsgd_classifier_test::exact_labels`, a hinge test, was dying on).
    //     The guard is unconditional at f64 for that reason, not because every
    //     loss needs the transcendental. ---
    crate::capability::guard_f64_transcendental::<F>("sgd_solve")?;

    let elem = size_of::<F>();
    let client = pool.client().clone();

    let max_iter = if params.max_iter == 0 {
        SGD_DEFAULT_MAX_ITER
    } else {
        params.max_iter
    };
    // WR-03: `batch > 1` is NOT sklearn-equivalent — the summed gradient is
    // averaged by `1/bsz` and the margin is not re-read mid-batch (see the
    // `SgdParams::batch_size` doc). Only `batch == 1` matches sklearn.
    let batch = params.batch_size.clamp(1, n);

    // The Bottou t0 (optimal schedule only); computed once before the loop.
    let t0 = optimal_t0(params.loss, params.alpha);
    let lid = loss_id(params.loss);
    let eps_f = f64_to_host::<F>(params.epsilon);

    // Device-resident solver state, acquired ONCE and reused every batch
    // (bounded allocation — the D-10 memory gate): the weights `w` (len d), the
    // intercept `bias` (len 1, returned to the caller as the fitted intercept),
    // a length-`batch` margin output `p` and gradient `g`.
    let w_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &vec![f64_to_host::<F>(0.0); d]);
    let bias_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &[f64_to_host::<F>(0.0)]);
    let p_handle = pool.acquire(batch * elem); // margin output, reused.
    let g_handle = pool.acquire(batch * elem); // per-sample gradient, reused.

    // Convergence tracking (tol > 0 only): a length-`batch` per-sample loss
    // buffer (reused every batch, like `p`/`g`) folded into a running length-1
    // `sumloss` epoch accumulator — sklearn's training-loss-plateau `tol` /
    // `n_iter_no_change` stop (MBSGD-PERF-WGPU; replaces a coefficient-delta
    // check that took far more epochs than sklearn to satisfy on data where
    // the loss had already plateaued). Both device-resident so the batch loop
    // stays synchronization-free.
    let track = params.tol > 0.0;
    let loss_handle = if track { Some(pool.acquire(batch * elem)) } else { None };

    // Cumulative-L1 state (L1/ElasticNet with alpha > 0 only): the device
    // per-coordinate applied-penalty `q[]` (sklearn `q`) and the host f64
    // mirror of the cumulative budget `u` (sklearn `u`). The mirror advances by
    // the SAME per-sample additions the kernel replays, so the `u_start` handed
    // to each launch stays in lock-step with the device trajectory.
    let l1_active = params.apply_l1 && params.l1_ratio > 0.0 && params.alpha > 0.0;
    let q_dev: Option<DeviceArray<ActiveRuntime, F>> = if l1_active {
        Some(DeviceArray::from_host(
            pool,
            &vec![f64_to_host::<F>(0.0); d],
        ))
    } else {
        None
    };
    let mut u_l1 = 0.0f64; // cumulative L1 penalty budget (sklearn wscale trick).

    // `t` counts SAMPLES consumed across epochs (sklearn's schedule clock).
    let mut t: u64 = 1;
    // sklearn's `best_loss`/`no_improvement_count` (`tol > 0` only) — carried
    // ACROSS epochs, unlike the per-epoch device buffers below.
    let mut best_loss = f64::INFINITY;
    let mut no_improvement_count: usize = 0;
    let n_iter_no_change = params.n_iter_no_change.max(1);

    let cube_block = 256u32;
    let dim = CubeDim {
        x: cube_block,
        y: 1,
        z: 1,
    };
    let one_count = CubeCount::Static(1, 1, 1);
    let one_dim = CubeDim { x: 1, y: 1, z: 1 };
    let d_count = CubeCount::Static((d as u32).div_ceil(cube_block), 1, 1);

    'epochs: for _epoch in 0..max_iter {
        // Zero the running epoch `sumloss` accumulator on device; read back
        // ONCE at epoch end (the only steady-state host sync).
        let sumloss_dev: Option<DeviceArray<ActiveRuntime, F>> = if track {
            Some(DeviceArray::from_host(pool, &[f64_to_host::<F>(0.0)]))
        } else {
            None
        };

        let mut start = 0usize;
        while start < n {
            let bsz = batch.min(n - start);
            let binv = 1.0 / bsz as f64;
            let b_count = CubeCount::Static((bsz as u32).div_ceil(cube_block), 1, 1);

            // --- Pass 1: margin p[i] = Σ_j x[(start+i),j]·w[j] + bias over the
            //     batch rows, addressed IN PLACE in the full design via the
            //     row_offset scalar (no per-batch slice upload). ---
            sgd_margin::launch::<F, ActiveRuntime>(
                &client,
                b_count.clone(),
                dim,
                unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) },
                unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) },
                unsafe { ArrayArg::from_raw_parts(bias_dev.handle().clone(), 1) },
                unsafe { ArrayArg::from_raw_parts(p_handle.clone(), bsz) },
                start as u32,
                bsz as u32,
                d as u32,
            );

            // --- Per-sample gradient g[i] = dloss(p_i, y_i) clipped ±1e12, on
            //     device (the dloss table lives in sgd_grad, selected by
            //     loss_id — no p[] readback / g[] upload). ---
            sgd_grad::launch::<F, ActiveRuntime>(
                &client,
                b_count.clone(),
                dim,
                unsafe { ArrayArg::from_raw_parts(p_handle.clone(), bsz) },
                unsafe { ArrayArg::from_raw_parts(y.handle().clone(), n) },
                unsafe { ArrayArg::from_raw_parts(g_handle.clone(), bsz) },
                start as u32,
                bsz as u32,
                lid,
                eps_f,
            );

            // --- Schedule eta for this batch (host f64; batch-start clock). ---
            let eta = schedule_eta(
                params.schedule,
                t,
                params.eta0,
                params.alpha,
                params.power_t,
                t0,
            );

            // --- Loss-plateau tracking (tol > 0 only): the per-sample loss
            //     VALUE `loss[i] = loss(p_i, y_i)` (distinct from `sgd_grad`'s
            //     derivative), folded into the running epoch `sumloss` — the
            //     sklearn training-loss `tol`/`n_iter_no_change` stop. ---
            if let (Some(loss_h), Some(sumloss)) = (&loss_handle, &sumloss_dev) {
                sgd_loss::launch::<F, ActiveRuntime>(
                    &client,
                    b_count.clone(),
                    dim,
                    unsafe { ArrayArg::from_raw_parts(p_handle.clone(), bsz) },
                    unsafe { ArrayArg::from_raw_parts(y.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(loss_h.clone(), bsz) },
                    start as u32,
                    bsz as u32,
                    lid,
                    eps_f,
                );
                sgd_sumloss::launch::<F, ActiveRuntime>(
                    &client,
                    one_count.clone(),
                    one_dim,
                    unsafe { ArrayArg::from_raw_parts(loss_h.clone(), bsz) },
                    unsafe { ArrayArg::from_raw_parts(sumloss.handle().clone(), 1) },
                    bsz as u32,
                );
            }

            // --- Pass 2: w[j] = (w[j] − eta·inv_b·Σ_i g[i]·x[i,j]) · l2_factor.
            //     CR-01: sklearn decays the weight scale ONCE PER SAMPLE inside
            //     its per-sample loop, so over a batch of `bsz` samples it
            //     shrinks by `(1 − eta·α·(1−l1_ratio))^bsz` — compounded here in
            //     host f64 and fused into the update kernel (order matches
            //     sklearn `_plain_sgd`: penalty shrink AFTER the gradient step).
            //     With alpha == 0 the factor is EXACTLY 1.0 (an exact
            //     multiplicative identity — the unpenalized path is unchanged). ---
            let l2_factor = if params.alpha > 0.0 {
                (1.0 - (1.0 - params.l1_ratio) * eta * params.alpha)
                    .max(0.0)
                    .powi(bsz as i32)
            } else {
                1.0
            };
            sgd_weight_update::launch::<F, ActiveRuntime>(
                &client,
                d_count.clone(),
                dim,
                unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) },
                unsafe { ArrayArg::from_raw_parts(g_handle.clone(), bsz) },
                unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) },
                f64_to_host::<F>(eta),
                f64_to_host::<F>(binv),
                f64_to_host::<F>(l2_factor),
                start as u32,
                d as u32,
                bsz as u32,
            );

            // --- Cumulative-L1 soft-shrink (sklearn `l1penalty`), on device.
            //     CR-02: the budget `u` advances and the soft-shrink applies
            //     ONCE PER SAMPLE (the kernel replays the `bsz` steps per
            //     coordinate, single-owner); the host f64 mirror advances by
            //     the same per-sample additions so `u_start` stays in
            //     lock-step across batches. ---
            if let Some(q) = &q_dev {
                let du = params.l1_ratio * eta * params.alpha;
                // One cube per coordinate, single unit (002-A: the loop-carried
                // multi-accumulator body needs the selecting-unit shape).
                sgd_l1_shrink::launch::<F, ActiveRuntime>(
                    &client,
                    CubeCount::Static(d as u32, 1, 1),
                    one_dim,
                    unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) },
                    unsafe { ArrayArg::from_raw_parts(q.handle().clone(), d) },
                    f64_to_host::<F>(u_l1),
                    f64_to_host::<F>(du),
                    d as u32,
                    bsz as u32,
                );
                // Advance the host f64 mirror the same way the kernel derives
                // its per-sample u (u_start + k·du, multiplicative — the
                // repeated-add form is a cpu-MLIR landmine in-kernel), so the
                // next batch's `u_start` matches the device trajectory on the
                // f64 path bit-for-bit.
                u_l1 += (bsz as f64) * du;
            }

            // --- Intercept step: bias -= eta·inv_b·Σ_i g_i (intercept_decay =
            //     1.0 dense, A3), device-resident single-unit fold. ---
            if params.fit_intercept {
                sgd_bias_update::launch::<F, ActiveRuntime>(
                    &client,
                    one_count.clone(),
                    one_dim,
                    unsafe { ArrayArg::from_raw_parts(g_handle.clone(), bsz) },
                    unsafe { ArrayArg::from_raw_parts(bias_dev.handle().clone(), 1) },
                    f64_to_host::<F>(eta),
                    f64_to_host::<F>(binv),
                    bsz as u32,
                );
            }

            t += bsz as u64;
            start += bsz;
        }

        // sklearn's training-loss-plateau stopping gate: ONE 1-scalar readback
        // per epoch (the only steady-state synchronization; absent entirely
        // when tol == 0) against `best_loss`/`no_improvement_count`, carried
        // across epochs exactly as `_plain_sgd` does.
        if let Some(sumloss) = sumloss_dev {
            let s_host = sumloss.to_host(pool);
            sumloss.release_into(pool);
            let sumloss64 = host_to_f64(s_host[0]);
            if sumloss64 > best_loss - params.tol * n as f64 {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }
            if sumloss64 < best_loss {
                best_loss = sumloss64;
            }
            if no_improvement_count >= n_iter_no_change {
                break 'epochs;
            }
        }
    }

    // Release the reusable per-batch scratch back to the pool.
    pool.release(p_handle, batch * elem);
    pool.release(g_handle, batch * elem);
    if let Some(loss_h) = loss_handle {
        pool.release(loss_h, batch * elem);
    }
    if let Some(q) = q_dev {
        q.release_into(pool);
    }

    Ok((w_dev, bias_dev))
}

/// Per-sample loss subgradient `dloss(p, y)` (RESEARCH §SGD-Math, the exact
/// sklearn `_sgd_fast` table). The HOST f64 reference for the device `sgd_grad`
/// kernel (which the kernel-vs-host gates in `sgd_test.rs` assert against), and
/// the `optimal`-schedule `t0` probe. `epsilon` is the epsilon-insensitive
/// margin (ignored by the other losses).
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

/// Per-sample loss VALUE `loss(p, y)` (as opposed to [`dloss`]'s derivative) —
/// sklearn's `_sgd_fast` `Loss.loss` table. The HOST f64 reference for the
/// device `sgd_loss` kernel (the `sgd_test.rs` kernel-vs-host gate) and for
/// [`sgd_host::loss_value_t`](super::sgd_host)'s bit-identity argument. Feeds
/// the sklearn training-loss-plateau `tol`/`n_iter_no_change` stop (replacing
/// a coefficient-delta check that, unlike sklearn, could take hundreds of
/// extra epochs to satisfy on data where the loss had already plateaued —
/// MBSGD-PERF-WGPU). `epsilon` is the epsilon-insensitive margin (ignored by
/// the other losses). The `Log` clip at `|z| = 18` mirrors sklearn's
/// `Log.loss` exactly (a `log(1+exp(-z))` without it overflows to `+inf` for a
/// badly-misclassified sample at typical `f32` magnitudes).
pub fn loss_value(loss: SgdLoss, p: f64, y: f64, epsilon: f64) -> f64 {
    match loss {
        SgdLoss::Hinge => (1.0 - p * y).max(0.0),
        SgdLoss::SquaredHinge => (1.0 - p * y).max(0.0).powi(2),
        SgdLoss::Log => {
            let z = p * y;
            if z > 18.0 {
                (-z).exp()
            } else if z < -18.0 {
                -z
            } else {
                (1.0 + (-z).exp()).ln()
            }
        }
        SgdLoss::SquaredError => 0.5 * (p - y).powi(2),
        SgdLoss::EpsilonInsensitive => ((p - y).abs() - epsilon).max(0.0),
        SgdLoss::SquaredEpsilonInsensitive => ((p - y).abs() - epsilon).max(0.0).powi(2),
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
    // IN-05: the `dloss` probe `epsilon` is INERT here — the `optimal` schedule is
    // only paired with the classifier losses (hinge / log / squared-hinge), none
    // of which read `epsilon` (matching sklearn's `optimal_init`, which likewise
    // ignores it). A named zero documents this rather than a bare magic `0.1`.
    const OPTIMAL_INIT_EPSILON: f64 = 0.0;
    let initial_eta0 =
        typw / dloss(loss, -typw, 1.0, OPTIMAL_INIT_EPSILON).abs().max(1.0);
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
