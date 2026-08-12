//! `device` — the execution-placement hyperparameter (DEVICE-PARAM-01).
//!
//! The parameter's whole promise is that it changes WHERE the work happens and
//! nothing else, so the load-bearing gate here is not "does `device='cpu'` take
//! the host arm" — it is **do the two arms agree on the answer**. A placement
//! knob that quietly changes the numbers is worse than no knob at all, because
//! a caller reaches for it to make things faster and has no reason to re-check
//! the fit.
//!
//! The second gate is that the parameter never LIES. `device` is a preference,
//! and some configurations have only one implementation (`solver='lsqr'` has no
//! host-slice ingress), so the estimator reports the arm that actually carried
//! the fit through `device_`. These tests pin that reporting, including the
//! case where the preference cannot be honoured.
//!
//! ```text
//! cargo test -p mlrs-algos --features cpu --test device_param_test
//! ```
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_algos::typestate::{Fit, Unfit};
use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{active_client, ActiveRuntime};

/// Counter-based splitmix64 — the workspace's deterministic design source.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn design(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();
    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();
    let mut sn = seed + 2;
    let y: Vec<f64> = (0..n)
        .map(|r| {
            let mut acc = 1.5 + 0.02 * uniform_pm1(&mut sn);
            for c in 0..d {
                acc += x[r * d + c] * coef[c];
            }
            acc
        })
        .collect();
    (x, y)
}

fn builder(device: Device, solver: RidgeSolver, positive: bool) -> Ridge<f64, Unfit> {
    Ridge::<f64>::builder()
        .alpha(0.7)
        .solver(solver)
        .positive(positive)
        .device(device)
        .build::<f64>()
        .expect("valid hyperparameters")
}

/// Fit through the SAME dispatch a real caller takes: branch on
/// `host_fit_applicable`, then either the host-slice ingress or the device one.
/// Returns `(coef, intercept, device_)`.
fn fit_dispatched(
    pool: &mut BufferPool<ActiveRuntime>,
    est: Ridge<f64, Unfit>,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
) -> (Vec<f64>, f64, &'static str) {
    if est.host_fit_applicable((n, d)) {
        let fitted = est
            .fit_from_host_slice(pool, x, y, (n, d), None)
            .expect("host fit");
        let arm = fitted.device();
        (fitted.coef(pool), fitted.intercept(pool), arm)
    } else {
        let xd = DeviceArray::from_host(pool, x);
        let yd = DeviceArray::from_host(pool, y);
        let fitted = est
            .fit(pool, &xd, Some(&yd), (n, d))
            .expect("device fit");
        let arm = fitted.device();
        (fitted.coef(pool), fitted.intercept(pool), arm)
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------------------
// The gate that matters: placement must not move the answer
// ---------------------------------------------------------------------------

#[test]
fn cpu_and_gpu_arms_agree_on_the_fit() {
    let (n, d) = (300usize, 8usize);
    let (x, y) = design(n, d, 21);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(active_client());

    let (coef_cpu, icpt_cpu, arm_cpu) = fit_dispatched(
        &mut pool,
        builder(Device::Cpu, RidgeSolver::Cholesky, false),
        &x,
        &y,
        n,
        d,
    );
    let (coef_gpu, icpt_gpu, arm_gpu) = fit_dispatched(
        &mut pool,
        builder(Device::Gpu, RidgeSolver::Cholesky, false),
        &x,
        &y,
        n,
        d,
    );

    assert_eq!(arm_cpu, "cpu", "device=Cpu did not take the host arm");
    assert_eq!(arm_gpu, "gpu", "device=Gpu did not take the device arm");

    // Two genuinely different implementations of the same normal equations, so
    // this is a tolerance rather than an equality — but a LOOSE one would make
    // the test vacuous, and 1e-9 is far inside the 1e-5 oracle gate.
    let err = max_abs_diff(&coef_cpu, &coef_gpu);
    assert!(
        err < 1e-9,
        "the two arms disagree on coef_ by {err:.3e} — a placement parameter \
         must not change the answer"
    );
    assert!(
        (icpt_cpu - icpt_gpu).abs() < 1e-9,
        "the two arms disagree on intercept_: {icpt_cpu} vs {icpt_gpu}"
    );
}

/// The `positive = true` arm has its own host/device split (`ridge_nnls` vs
/// `nonnegative_cd`), so it needs its own agreement gate rather than riding on
/// the `cholesky` one.
#[test]
fn cpu_and_gpu_arms_agree_with_positive() {
    let (n, d) = (200usize, 6usize);
    let (x, y) = design(n, d, 33);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(active_client());

    let (coef_cpu, _, arm_cpu) = fit_dispatched(
        &mut pool,
        builder(Device::Cpu, RidgeSolver::Auto, true),
        &x,
        &y,
        n,
        d,
    );
    let (coef_gpu, _, arm_gpu) = fit_dispatched(
        &mut pool,
        builder(Device::Gpu, RidgeSolver::Auto, true),
        &x,
        &y,
        n,
        d,
    );
    assert_eq!(arm_cpu, "cpu");
    assert_eq!(arm_gpu, "gpu");
    assert!(coef_cpu.iter().all(|c| *c >= -1e-12), "bound violated on cpu");
    assert!(coef_gpu.iter().all(|c| *c >= -1e-12), "bound violated on gpu");
    let err = max_abs_diff(&coef_cpu, &coef_gpu);
    assert!(err < 1e-6, "positive arms disagree by {err:.3e}");
}

// ---------------------------------------------------------------------------
// The parameter must never lie about what ran
// ---------------------------------------------------------------------------

/// `solver='lsqr'` has no host-slice ingress, so `device = Cpu` CANNOT be
/// honoured. The fit must still run, and `device_` must say `"gpu"` — the
/// case the fitted attribute exists for.
#[test]
fn an_unhonourable_preference_is_reported_not_faked() {
    let (n, d) = (150usize, 5usize);
    let (x, y) = design(n, d, 41);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(active_client());

    let est = builder(Device::Cpu, RidgeSolver::Lsqr, false);
    assert!(
        !est.host_fit_applicable((n, d)),
        "lsqr has no host-slice ingress; device=Cpu must not claim one"
    );
    let (coef, _, arm) = fit_dispatched(&mut pool, est, &x, &y, n, d);
    assert_eq!(
        arm, "gpu",
        "device_ must report the arm that RAN, not the one that was asked for"
    );
    assert!(coef.iter().all(|c| c.is_finite()), "lsqr fit produced non-finite coef");
}

/// `Auto` must reproduce the pre-parameter behaviour exactly — the heuristic,
/// including its `MLRS_RIDGE_GRAM_HOST` A/B flag. If this drifts, every perf
/// probe and bench in the repo silently changes meaning.
#[test]
fn auto_preserves_the_heuristic_and_its_abflag() {
    let (n, d) = (256usize, 8usize);
    let est = || builder(Device::Auto, RidgeSolver::Cholesky, false);

    let forced_host = {
        let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
        est().host_fit_applicable((n, d))
    };
    let forced_device = {
        let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
        est().host_fit_applicable((n, d))
    };
    assert!(forced_host, "Auto stopped honouring the host A/B flag");
    assert!(!forced_device, "Auto stopped honouring the device A/B flag");
}

/// An EXPLICIT `device` outranks the A/B flag, which is what makes the
/// parameter reproducible: a stray `MLRS_*` in the environment must not move a
/// fit the caller pinned.
#[test]
fn an_explicit_device_ignores_the_abflag() {
    let (n, d) = (256usize, 8usize);
    let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
    assert!(
        builder(Device::Cpu, RidgeSolver::Cholesky, false).host_fit_applicable((n, d)),
        "device=Cpu was overridden by MLRS_RIDGE_GRAM_HOST=0"
    );
    let _g2 = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
    assert!(
        !builder(Device::Gpu, RidgeSolver::Cholesky, false).host_fit_applicable((n, d)),
        "device=Gpu was overridden by MLRS_RIDGE_GRAM_HOST=1"
    );
}

// ---------------------------------------------------------------------------
// Parsing and defaults
// ---------------------------------------------------------------------------

#[test]
fn device_parses_and_round_trips() {
    for (name, want) in [
        ("auto", Device::Auto),
        ("cpu", Device::Cpu),
        ("gpu", Device::Gpu),
    ] {
        assert_eq!(Device::from_name(name), Some(want));
        assert_eq!(want.name(), name, "name() must round-trip from_name()");
    }
    assert_eq!(Device::from_name("cuda"), None);
    assert_eq!(Device::from_name("CPU"), None, "the parse is case-sensitive");
}

#[test]
fn default_is_auto_and_the_builder_agrees() {
    assert_eq!(Device::default(), Device::Auto);
    // D-08: the builder's defaults are re-derived from `Ridge::new`, so adding
    // a field to one and not the other is exactly the drift this catches.
    assert!(Ridge::<f64>::new()
        .hyperparams_eq(&Ridge::<f64>::builder().build::<f64>().expect("defaults")));
}
