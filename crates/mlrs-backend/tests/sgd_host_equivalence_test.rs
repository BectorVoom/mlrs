//! MBSGD-PERF-CPU — the cpu HOST arm of `sgd_solve` must be a BITWISE replay
//! of the device kernel pipeline, not merely a close one.
//!
//! SGD is a sequential recurrence: sample `i + 1`'s margin is read off the
//! weights sample `i` wrote, so a single last-ULP difference compounds over
//! `n · max_iter` steps and moves the fitted iterate macroscopically. A
//! tolerance test would only catch that by luck. These tests therefore compare
//! the two arms' `(coef, intercept)` BIT FOR BIT (`to_bits`), across every
//! loss family, both schedules, both float types, the L1 / ElasticNet
//! cumulative-shrink path, the `tol > 0` convergence-tracking path (whose
//! maxima the host arm reduces LANE-SPLIT rather than serially), and
//! `batch_size > 1`.
//!
//! The arms are selected through the public prim with the `abflag`
//! thread-local override (never `std::env::set_var` — see the `abflag` module
//! doc on the `environ` data race and silently-vacuous kernel-agreement
//! assertions).
//!
//! cpu-only: `sgd_host` is gated to the cpu backend, so on wgpu/CUDA/ROCm both
//! arms would be the same device path and the comparison would be vacuous.
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)]` module.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::sgd::{sgd_solve, SgdLoss, SgdParams, SgdSchedule};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The host arm only exists on cpu; elsewhere this comparison is vacuous.
fn skip_off_cpu() -> bool {
    if capability::active_backend_name() != "cpu" {
        eprintln!("sgd host-arm equivalence is cpu-only; skipping");
        return true;
    }
    false
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("sgd tests are f32/f64 only"),
    }
}

/// Raw bit pattern of an `F` value, for the exact-equality assertions.
fn bits<F: Pod>(v: F) -> u64 {
    match std::mem::size_of::<F>() {
        4 => bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)).to_bits() as u64,
        8 => bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)).to_bits(),
        _ => unreachable!("sgd tests are f32/f64 only"),
    }
}

/// A centred design with a ±1 (classification) / continuous (regression)
/// target — both are exercised, the loss decides which reading applies.
fn make_data<F: Pod>(n: usize, d: usize, seed: u64) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut s = seed;
    let mut x = Vec::with_capacity(n * d);
    let mut y_pm1 = Vec::with_capacity(n);
    let mut y_reg = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            row.push(uniform01(&mut s) * 2.0 - 1.0);
        }
        let t = 1.5 * row[0] - 0.75 * row[1 % d] + 0.25 * uniform01(&mut s);
        y_pm1.push(f64_to::<F>(if t >= 0.0 { 1.0 } else { -1.0 }));
        y_reg.push(f64_to::<F>(t));
        x.extend(row.into_iter().map(f64_to::<F>));
    }
    (x, y_pm1, y_reg)
}

/// Run `sgd_solve` on one arm (`host = false` forces the device path via the
/// `MLRS_SGD_HOST=0` knob) and return the fitted `(coef, intercept)`.
fn solve_arm<F>(x: &[F], y: &[F], n: usize, d: usize, params: &SgdParams, host: bool) -> (Vec<F>, F)
where
    F: Float + CubeElement + Pod,
{
    // `MLRS_SGD_WIDE_DOT` is a DELIBERATE reassociation of the margin (see
    // `sgd_host::wide_dot`); pin it off so an ambient environment cannot turn
    // this bit-identity gate into a failure — or, worse, quietly redefine what
    // the "host arm" being asserted here even is.
    let _wide = abflag::clear("MLRS_SGD_WIDE_DOT");
    let _guard = if host {
        abflag::clear("MLRS_SGD_HOST")
    } else {
        abflag::force("MLRS_SGD_HOST", "0")
    };
    let client = runtime::active_client();
    let mut pool = BufferPool::<ActiveRuntime>::new(client);
    let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, x);
    let y_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, y);
    let (coef, intercept) =
        sgd_solve::<F>(&mut pool, &x_dev, &y_dev, (n, d), params).expect("sgd_solve");
    let c = coef.to_host(&mut pool);
    let b = intercept.to_host(&mut pool)[0];
    (c, b)
}

/// Assert the two arms agree BIT FOR BIT for one parameter set.
fn assert_arms_agree<F>(label: &str, n: usize, d: usize, params: &SgdParams)
where
    F: Float + CubeElement + Pod,
{
    let (x, y_pm1, y_reg) = make_data::<F>(n, d, 0xC0FF_EE12);
    let y: &[F] = match params.loss {
        SgdLoss::Hinge | SgdLoss::Log | SgdLoss::SquaredHinge => &y_pm1,
        _ => &y_reg,
    };

    let (dev_c, dev_b) = solve_arm::<F>(&x, y, n, d, params, false);
    let (host_c, host_b) = solve_arm::<F>(&x, y, n, d, params, true);

    assert_eq!(dev_c.len(), host_c.len(), "{label}: coef length");
    for j in 0..dev_c.len() {
        assert_eq!(
            bits(host_c[j]),
            bits(dev_c[j]),
            "{label}: coef[{j}] host arm diverged from the device arm \
             (host={:e}, device={:e})",
            f64::from_bits(widen(host_c[j])),
            f64::from_bits(widen(dev_c[j])),
        );
    }
    assert_eq!(
        bits(host_b),
        bits(dev_b),
        "{label}: intercept host arm diverged from the device arm"
    );
}

/// Widen an `F` bit pattern to the f64 bit pattern of the same value, so the
/// failure message prints a readable number for either precision.
fn widen<F: Pod>(v: F) -> u64 {
    match std::mem::size_of::<F>() {
        4 => (*bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64).to_bits(),
        8 => bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)).to_bits(),
        _ => unreachable!("sgd tests are f32/f64 only"),
    }
}

/// The pinned-oracle shape: `tol = 0`, `batch = 1`, L2, optimal schedule.
fn base_params(loss: SgdLoss) -> SgdParams {
    SgdParams {
        loss,
        schedule: SgdSchedule::Optimal,
        alpha: 1e-4,
        l1_ratio: 0.15,
        apply_l1: false,
        fit_intercept: true,
        eta0: 0.01,
        power_t: 0.5,
        epsilon: 0.1,
        batch_size: 1,
        max_iter: 12,
        tol: 0.0,
    }
}

const N: usize = 64;
const D: usize = 5;

/// Every loss family, both float types — the `dloss` table must round the same
/// way in `F` on both arms (including the `F::new(1e12_f32)` clip, which at f64
/// is `999999995904.0` and NOT `1e12`).
#[test]
fn host_arm_matches_device_all_losses() {
    if skip_off_cpu() {
        return;
    }
    let losses = [
        ("hinge", SgdLoss::Hinge),
        ("log", SgdLoss::Log),
        ("squared_hinge", SgdLoss::SquaredHinge),
        ("squared_error", SgdLoss::SquaredError),
        ("eps_insensitive", SgdLoss::EpsilonInsensitive),
        ("sq_eps_insensitive", SgdLoss::SquaredEpsilonInsensitive),
    ];
    for (name, loss) in losses {
        assert_arms_agree::<f32>(&format!("f32/{name}"), N, D, &base_params(loss));
        assert_arms_agree::<f64>(&format!("f64/{name}"), N, D, &base_params(loss));
    }
}

/// The non-`optimal` schedules and the no-intercept path.
#[test]
fn host_arm_matches_device_schedules() {
    if skip_off_cpu() {
        return;
    }
    for (name, schedule) in [
        ("constant", SgdSchedule::Constant),
        ("invscaling", SgdSchedule::InvScaling),
        ("adaptive", SgdSchedule::Adaptive),
    ] {
        let mut p = base_params(SgdLoss::Hinge);
        p.schedule = schedule;
        assert_arms_agree::<f32>(&format!("f32/{name}"), N, D, &p);
        assert_arms_agree::<f64>(&format!("f64/{name}"), N, D, &p);

        p.fit_intercept = false;
        assert_arms_agree::<f64>(&format!("f64/{name}/no-intercept"), N, D, &p);
    }
}

/// The cumulative-L1 shrink (`sgd_l1_shrink`): the host arm must derive `u`
/// from the sample counter the same way the kernel does, and advance the same
/// `u_start` mirror across batches.
#[test]
fn host_arm_matches_device_l1_paths() {
    if skip_off_cpu() {
        return;
    }
    for (name, l1_ratio) in [("l1", 1.0), ("elasticnet", 0.15)] {
        let mut p = base_params(SgdLoss::Hinge);
        p.apply_l1 = true;
        p.l1_ratio = l1_ratio;
        // A penalty large enough that the soft-shrink actually clamps.
        p.alpha = 1e-2;
        assert_arms_agree::<f32>(&format!("f32/{name}"), N, D, &p);
        assert_arms_agree::<f64>(&format!("f64/{name}"), N, D, &p);
    }
}

/// `tol > 0`: the host arm FUSES the WR-02 start-of-batch delta into its update
/// loop instead of snapshotting `w`, so it must stop on exactly the same epoch
/// as the device `sgd_copy` + `sgd_delta_max` pair — and with L1 active it must
/// fall back to the explicit snapshot (the weights move again after the step).
#[test]
fn host_arm_matches_device_tol_tracking() {
    if skip_off_cpu() {
        return;
    }
    let mut p = base_params(SgdLoss::Hinge);
    p.tol = 1e-3;
    p.max_iter = 40;
    assert_arms_agree::<f32>("f32/tol", N, D, &p);
    assert_arms_agree::<f64>("f64/tol", N, D, &p);

    p.apply_l1 = true;
    p.alpha = 1e-2;
    assert_arms_agree::<f64>("f64/tol+l1", N, D, &p);
}

/// `batch_size > 1` (the documented non-sklearn-equivalent mode) still has to
/// replay identically, including the ragged final batch and the compounded
/// per-sample L2 factor.
#[test]
fn host_arm_matches_device_minibatch() {
    if skip_off_cpu() {
        return;
    }
    // 64 rows / batch 7 ⇒ a ragged tail batch of 1, which takes the peeled
    // single-sample path on the host arm.
    for batch in [2usize, 7, 64] {
        let mut p = base_params(SgdLoss::SquaredHinge);
        p.batch_size = batch;
        assert_arms_agree::<f32>(&format!("f32/batch{batch}"), N, D, &p);
        assert_arms_agree::<f64>(&format!("f64/batch{batch}"), N, D, &p);

        // The same shape with tracking on — the batched path keeps the
        // explicit snapshot.
        p.tol = 1e-4;
        p.max_iter = 30;
        assert_arms_agree::<f64>(&format!("f64/batch{batch}/tol"), N, D, &p);
    }
}
