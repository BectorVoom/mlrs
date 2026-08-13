//! `RidgeCV`'s host and device GCV arms must produce the same fit (RIDGECV-02).
//!
//! ## What this gates, and what it deliberately does not
//! A placement knob that changes the numbers is worse than no knob: a caller
//! reaches for `device='gpu'` to go faster and has no reason to re-check the
//! fit. So the contract is AGREEMENT, not selection — the same one
//! `test_device_param.py::cpu_and_gpu_arms_agree_on_the_fit` holds for the
//! estimators that already had the parameter.
//!
//! Agreement is to ~`ε·κ`, not bit-for-bit: both arms evaluate the same `f64`
//! expressions but fold in different orders (the device per row block, the host
//! per worker chunk). `prims::ridge_gcv`'s module docs state that contract; this
//! file is the gate on it.
//!
//! All FOUR outputs are compared, not just the scores. A sweep that got the LOO
//! denominator right and the prediction re-scale wrong passes a scores-only
//! check on an unweighted fixture, and a coefficient block indexed
//! `(a·d + j)·n_y + t` the wrong way round passes both.
//!
//! ## This suite SKIPS on cpu and wgpu, and that is not a gap here
//! `gcv_device_possible` is `false` on both (no fused Gram kernel / no `f64`
//! device kernels), so on the two backends CI gates on there is only one arm and
//! nothing to compare. The launch sites are kept EXECUTABLE there by
//! `mlrs-backend/tests/ridge_gcv_test.rs`, which calls the prim directly against
//! naive references — that file is the cpu-CI half of this pair.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use mlrs_algos::linear::ridge_cv::{gcv_device_arm, ridge_gcv, ridge_gcv_auto, GcvFit, GcvMode};
use mlrs_backend::device::Device;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::ridge_gcv::gcv_device_possible;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_backend::{abflag, capability};

/// Deterministic pseudo-random source (splitmix64).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)`.
fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A well-conditioned design with a real signal, offset means and a spread of
/// weights — the configuration in which every branch of the sweep is live.
fn make_case(n: usize, d: usize, n_y: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..n * d)
        .map(|k| uniform_pm1(&mut s) + 1.0 + (k % d) as f64)
        .collect();
    let beta: Vec<f64> = (0..d * n_y).map(|_| uniform_pm1(&mut s)).collect();
    let mut y = vec![0.0f64; n * n_y];
    for i in 0..n {
        for t in 0..n_y {
            let mut acc = 2.0 + t as f64;
            for j in 0..d {
                acc += x[i * d + j] * beta[j * n_y + t];
            }
            y[i * n_y + t] = acc + 0.1 * uniform_pm1(&mut s);
        }
    }
    let w: Vec<f64> = (0..n)
        .map(|_| 0.25 + 2.0 * (uniform_pm1(&mut s) + 1.0))
        .collect();
    (x, y, w)
}

/// Relative-or-absolute agreement between the two arms.
///
/// `1e-9` is the `f64` band `mlrs-device-param` records for arm agreement; the
/// engine is `f64` on both arms whatever the estimator's `F`, so the band does
/// not depend on the fixture's width.
fn assert_arms_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-9 * b.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "{label}[{i}]: device {a}, host {b} (tol {tol})"
        );
    }
}

fn assert_fits_close(dev: &GcvFit, host: &GcvFit, label: &str) {
    assert_arms_close(&dev.scores, &host.scores, &format!("{label} scores"));
    assert_arms_close(&dev.coefs, &host.coefs, &format!("{label} coefs"));
    assert_arms_close(
        &dev.cv_values,
        &host.cv_values,
        &format!("{label} cv_values"),
    );
    assert_arms_close(&dev.x_offset, &host.x_offset, &format!("{label} x_offset"));
    assert_arms_close(&dev.y_offset, &host.y_offset, &format!("{label} y_offset"));
    assert_eq!(dev.route, host.route, "{label}: route");
}

/// Is there a second arm on this backend? Prints WHY it skips, so a run that
/// asserted nothing cannot be mistaken for a run that passed.
fn device_arm_or_skip(d: usize, what: &str) -> bool {
    if !gcv_device_possible(d) {
        println!(
            "ridge_cv device {what}: SKIPPED (no device GCV arm on backend={} at d={d})",
            capability::active_backend_name()
        );
        return false;
    }
    true
}

/// The whole parameter surface, one case at a time: each configuration runs the
/// device arm and the host arm on the same fixture and demands they agree.
#[allow(clippy::too_many_arguments)]
fn run_agreement(
    n: usize,
    d: usize,
    n_y: usize,
    weighted: bool,
    fit_intercept: bool,
    want_predictions: bool,
    store_cv_values: bool,
    alphas: &[f64],
    seed: u64,
) {
    if !device_arm_or_skip(d, "agreement") {
        return;
    }
    let (x, y, w) = make_case(n, d, n_y, seed);
    let sw: Option<&[f64]> = if weighted { Some(&w) } else { None };
    let label = format!(
        "n={n} d={d} n_y={n_y} weighted={weighted} fi={fit_intercept} \
         pred={want_predictions} store={store_cv_values} n_alphas={}",
        alphas.len()
    );

    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let (device_fit, arm) = ridge_gcv_auto::<f64>(
        &mut pool,
        &x,
        &y,
        n,
        d,
        n_y,
        sw,
        alphas,
        fit_intercept,
        GcvMode::Auto,
        want_predictions,
        store_cv_values,
        Device::Gpu,
    )
    .expect("device arm must fit");
    // `mlrs-bench-verify-knob-is-live`: a comparison against an arm that
    // silently declined is host-against-host and passes vacuously.
    assert_eq!(
        arm, "gpu",
        "{label}: asked for the device arm and did not get it"
    );

    let host_fit = ridge_gcv::<f64>(
        &x,
        &y,
        n,
        d,
        n_y,
        sw,
        alphas,
        fit_intercept,
        GcvMode::Auto,
        want_predictions,
        store_cv_values,
    )
    .expect("host arm must fit");

    assert_fits_close(&device_fit, &host_fit, &label);
}

const ALPHAS: [f64; 4] = [0.01, 0.1, 1.0, 10.0];

#[test]
fn arms_agree_on_the_default_configuration() {
    run_agreement(600, 8, 1, false, true, false, false, &ALPHAS, 0xD00D_0001);
}

#[test]
fn arms_agree_with_sample_weight() {
    run_agreement(600, 8, 1, true, true, false, false, &ALPHAS, 0xD00D_0002);
}

#[test]
fn arms_agree_without_an_intercept() {
    run_agreement(600, 8, 1, false, false, false, false, &ALPHAS, 0xD00D_0003);
    run_agreement(600, 8, 1, true, false, false, false, &ALPHAS, 0xD00D_0004);
}

/// `scoring != None` is the arm that returns rescaled LOO PREDICTIONS instead of
/// squared errors, and under weights it also divides them back out by `√wᵢ`.
#[test]
fn arms_agree_on_the_prediction_output() {
    run_agreement(500, 8, 1, false, true, true, false, &ALPHAS, 0xD00D_0005);
    run_agreement(500, 8, 1, true, true, true, false, &ALPHAS, 0xD00D_0006);
}

/// `store_cv_results=True` fills the same `n × n_alphas × n_y` buffer while
/// still scoring in Rust — the one configuration where BOTH outputs are live.
#[test]
fn arms_agree_with_store_cv_results() {
    run_agreement(500, 8, 1, false, true, false, true, &ALPHAS, 0xD00D_0007);
}

/// Multi-target is what `alpha_per_target` reduces over, so the per-alpha
/// coefficient block and the per-`(alpha, target)` scores both have to survive.
#[test]
fn arms_agree_on_a_multi_target_fit() {
    run_agreement(400, 6, 3, false, true, false, true, &ALPHAS, 0xD00D_0008);
    run_agreement(400, 6, 2, true, true, false, false, &ALPHAS, 0xD00D_0009);
}

/// A single alpha and a long grid: the sweep's alpha loop is strided over units,
/// so `n_alphas = 1` leaves most of a cube idle and `n_alphas > CUBE_DIM` makes
/// each unit own several alphas. Both are off the common path.
#[test]
fn arms_agree_across_grid_lengths() {
    run_agreement(400, 6, 1, false, true, false, false, &[1.0], 0xD00D_000A);
    let long: Vec<f64> = (0..100).map(|i| 0.01 * 1.1f64.powi(i)).collect();
    run_agreement(400, 6, 1, false, true, false, false, &long, 0xD00D_000B);
}

/// `d` at the sweep's shared-tile ceiling, and a `d` that is not a multiple of
/// the 64-wide cube.
#[test]
fn arms_agree_at_awkward_feature_counts() {
    run_agreement(400, 65, 1, false, true, false, false, &ALPHAS, 0xD00D_000C);
    run_agreement(600, 256, 1, false, true, false, false, &ALPHAS, 0xD00D_000D);
}

/// An `f32` estimator still runs an `f64` engine on both arms — the design is
/// widened on the device rather than the accumulation being narrowed.
#[test]
fn arms_agree_for_an_f32_design() {
    let (n, d, n_y) = (600usize, 8usize, 1usize);
    if !device_arm_or_skip(d, "f32 agreement") {
        return;
    }
    let (x64, y64, _) = make_case(n, d, n_y, 0xD00D_000E);
    let x: Vec<f32> = x64.iter().map(|v| *v as f32).collect();
    let y: Vec<f32> = y64.iter().map(|v| *v as f32).collect();

    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let (device_fit, arm) = ridge_gcv_auto::<f32>(
        &mut pool,
        &x,
        &y,
        n,
        d,
        n_y,
        None,
        &ALPHAS,
        true,
        GcvMode::Auto,
        false,
        true,
        Device::Gpu,
    )
    .expect("device arm must fit");
    assert_eq!(arm, "gpu");
    let host_fit = ridge_gcv::<f32>(
        &x,
        &y,
        n,
        d,
        n_y,
        None,
        &ALPHAS,
        true,
        GcvMode::Auto,
        false,
        true,
    )
    .expect("host arm must fit");
    assert_fits_close(&device_fit, &host_fit, "f32 design");
}

/// The `n ≤ d` route has ONE arm — its cost is a serial `O(n³)` `sym_eig` no
/// kernel here addresses — so `device='gpu'` must fall back and SAY it fell
/// back rather than reporting an arm that never ran.
#[test]
fn the_wide_route_reports_the_host_arm_it_actually_ran() {
    let (n, d, n_y) = (40usize, 120usize, 1usize);
    let (x, y, _) = make_case(n, d, n_y, 0xD00D_000F);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let (_, arm) = ridge_gcv_auto::<f64>(
        &mut pool,
        &x,
        &y,
        n,
        d,
        n_y,
        None,
        &ALPHAS,
        true,
        GcvMode::Auto,
        false,
        false,
        Device::Gpu,
    )
    .expect("wide route must fit");
    assert_eq!(
        arm, "cpu",
        "the n <= d route has no device arm, so it must report the host one"
    );
    assert!(
        !gcv_device_arm(Device::Gpu, GcvMode::Auto, n, d, ALPHAS.len(), n_y),
        "gcv_device_arm must agree with what ridge_gcv_auto ran"
    );
}

/// `device='cpu'`/`'gpu'` are reproducible: an EXPLICIT preference never
/// consults the A/B flag, which is what keeps every perf probe in this repo
/// (all of which force through `abflag`) meaningful.
#[test]
fn an_explicit_device_ignores_the_abflag() {
    let (n, d, n_y) = (400usize, 8usize, 1usize);
    let has_device = gcv_device_possible(d);

    let _g = abflag::force("MLRS_RIDGECV_DEVICE", "0");
    assert_eq!(
        gcv_device_arm(Device::Gpu, GcvMode::Auto, n, d, ALPHAS.len(), n_y),
        has_device,
        "device='gpu' must not be turned off by the flag"
    );
    assert!(
        !gcv_device_arm(Device::Auto, GcvMode::Auto, n, d, ALPHAS.len(), n_y),
        "device='auto' must obey the flag"
    );
    drop(_g);

    let _g = abflag::force("MLRS_RIDGECV_DEVICE", "1");
    assert!(
        !gcv_device_arm(Device::Cpu, GcvMode::Auto, n, d, ALPHAS.len(), n_y),
        "device='cpu' must not be turned on by the flag"
    );
    assert_eq!(
        gcv_device_arm(Device::Auto, GcvMode::Auto, n, d, ALPHAS.len(), n_y),
        has_device,
        "device='auto' must obey the flag where the arm is legal"
    );
}

/// The capability half is NOT overridable: forcing the flag on a backend
/// without the arm must still decline, because an override there is a crash and
/// not a slowdown (`mlrs-device-param`'s rule).
#[test]
fn the_flag_cannot_force_an_arm_that_does_not_exist() {
    let _g = abflag::force("MLRS_RIDGECV_DEVICE", "1");
    // `d` past the sweep's shared-memory tile has no kernel on ANY backend.
    assert!(
        !gcv_device_arm(Device::Auto, GcvMode::Auto, 10_000, 4096, 3, 1),
        "d beyond the sweep's tile must have no device arm"
    );
    assert!(
        !gcv_device_arm(Device::Gpu, GcvMode::Auto, 10_000, 4096, 3, 1),
        "device='gpu' must not override a capability gate"
    );
}
