//! `RidgeCV` (RIDGECV-02) — the host-vs-DEVICE A/B ladder for the GCV engine.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features rocm \
//!   --test ridge_cv_device_perf_test -- --ignored --nocapture
//! ```
//!
//! This is the measurement `prims::ridge_gcv::gcv_device_preferred`'s default is
//! allowed to move on, and nothing else. It answers one question per ladder:
//!
//! 1. `device_vs_host_shape_ladder` — at a fixed alpha grid, where does the
//!    `O(n·d²)` work stop being worth the `O(n·d)` upload?
//! 2. `device_vs_host_alphas_ladder` — `len(alphas)` is the parameter this
//!    estimator is USED to sweep, and it is the one that moves the arithmetic
//!    per uploaded element (`2·d + n_alphas·(n_y+2)`). If the device arm wins
//!    anywhere it should win FIRST here, and the ratio should climb with the
//!    grid.
//! 3. `device_phase_attribution` — inside a device fit, how much is the upload,
//!    how much the normal equations, how much the sweep. A ratio without this is
//!    a number you cannot act on: `BayesianRidge` measured 92% of a device fit
//!    in the transfer and the arm's whole default turned on that fact.
//!
//! ## The arm is forced, and the force is VERIFIED
//! Both arms are driven through `MLRS_RIDGECV_DEVICE` and each timed call
//! asserts the arm it actually got (`ridge_gcv_auto`'s second return value). A
//! sweep against a gate that silently declined is host-against-host and flat by
//! construction — `mlrs-bench-verify-knob-is-live` is written down because that
//! has happened here before.
//!
//! ## Interleaved min-of-N, on purpose
//! Each rung times the two arms alternately rather than one ladder after the
//! other, so a machine that gets busy part-way through penalizes both arms
//! rather than inverting the verdict (`mlrs-hgb-cpu-bench-caveat`). `MLRS_
//! RIDGECV_REPS` overrides the repetition count.
//!
//! ## Contention is REPORTED, because it moves the verdict and not just the
//! ## variance
//! The host arm is a 16-thread `std::thread::scope` sweep and the device arm is
//! one host thread plus a GPU, so a busy machine penalizes the HOST arm far more
//! — i.e. contention flatters the device and the error has a SIGN. Each ladder
//! therefore samples `/proc/stat` around itself, subtracts its own cpu time, and
//! prints the share of the machine other processes took, with a banner past
//! [`FOREIGN_LIMIT`] (`scripts/bench_ridge_cv.py`'s guard, in Rust). A run whose
//! banner fired is not a measurement, and the ladders in
//! `prims::ridge_gcv::gcv_device_preferred` are only ever updated from runs
//! whose banner did not.
//!
//! `#[ignore]`d: a wall-clock measurement, never a gate.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge_cv::{gcv_device_arm, ridge_gcv_auto, GcvMode};
use mlrs_backend::capability;
use mlrs_backend::device::Device;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::ridge_gcv::{gcv_device_possible, GcvDevice};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Share of the machine OTHER processes may take during a ladder before it is
/// reported as untrusted. Same 10% `scripts/bench_ridge_cv.py` uses.
const FOREIGN_LIMIT: f64 = 0.10;

/// `(wall, system busy seconds, this process's cpu seconds)` right now.
///
/// The quantity that matters is FOREIGN cpu, not load average: a 16-thread
/// benchmark on a 16-core box IS a load of ~16, so a load-average guard fires on
/// every honest run. Busy-minus-own divided by `wall × cores` is the share
/// somebody else took, which is exactly what distorts a parallel-vs-GPU ratio.
fn cpu_sample() -> (f64, f64, f64) {
    let stat = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let parts: Vec<f64> = stat
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    // USER_HZ is 100 on every Linux this repo runs on; the ratio below cancels
    // it anyway, since `own` is measured in seconds and only their DIFFERENCE
    // over the same window is used.
    let ticks = 100.0;
    let total: f64 = parts.iter().sum::<f64>() / ticks;
    let idle = (parts.get(3).copied().unwrap_or(0.0) + parts.get(4).copied().unwrap_or(0.0))
        / ticks;
    let own = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            // utime + stime are fields 14 and 15 AFTER the comm field, which can
            // itself contain spaces — split at the last ')' the way procps does.
            let rest = s.rsplit_once(')')?.1;
            let f: Vec<f64> = rest
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            Some((f.get(11).copied()? + f.get(12).copied()?) / ticks)
        })
        .unwrap_or(0.0);
    (
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        total - idle,
        own,
    )
}

/// Print the foreign-cpu share for the window between two [`cpu_sample`]s, with
/// a banner when it is high enough to have moved the ratio.
fn report_foreign(before: (f64, f64, f64), label: &str) {
    let after = cpu_sample();
    let wall = after.0 - before.0;
    let cores = std::thread::available_parallelism()
        .map(|v| v.get() as f64)
        .unwrap_or(1.0);
    if wall <= 0.0 {
        return;
    }
    let share = ((after.1 - before.1) - (after.2 - before.2)).max(0.0) / (wall * cores);
    println!("  foreign cpu during {label}: {:.1}%", share * 100.0);
    if share > FOREIGN_LIMIT {
        println!(
            "  *** UNTRUSTED: other processes took {:.0}% of the machine (limit \
             {:.0}%). This compares a 16-thread HOST arm with a GPU arm, so \
             contention FLATTERS the device — the error has a sign. Re-run on a \
             quiet machine before believing these ratios. ***",
            share * 100.0,
            FOREIGN_LIMIT * 100.0
        );
    }
}

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

fn make_regression(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();
    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();
    let mut sn = seed + 2;
    let y: Vec<f64> = (0..n)
        .map(|r| {
            let mut acc = 0.5 + 0.01 * uniform_pm1(&mut sn);
            for c in 0..d {
                acc += x[r * d + c] * coef[c];
            }
            acc
        })
        .collect();
    (x, y)
}

fn reps() -> usize {
    std::env::var("MLRS_RIDGECV_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

fn logspace(lo: f64, hi: f64, k: usize) -> Vec<f64> {
    if k == 1 {
        return vec![10f64.powf(lo)];
    }
    (0..k)
        .map(|i| 10f64.powf(lo + (hi - lo) * i as f64 / (k - 1) as f64))
        .collect()
}

/// One `ridge_gcv_auto` fit on the requested arm, asserting the arm it got.
///
/// The design is uploaded INSIDE the timed region on the device arm, because
/// that is what the estimator's host-slice ingress actually pays.
fn fit_once(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    alphas: &[f64],
    device: Device,
) {
    let (_, arm) = ridge_gcv_auto::<f64>(
        pool,
        x,
        y,
        n,
        d,
        1,
        None,
        alphas,
        true,
        GcvMode::Auto,
        false,
        false,
        device,
    )
    .expect("fit");
    let want = if device == Device::Gpu { "gpu" } else { "cpu" };
    assert_eq!(
        arm, want,
        "asked for device={:?} and the fit ran on {arm}",
        device
    );
}

/// Interleaved min-of-N for the two arms: `(host_ms, device_ms)`.
fn measure_pair(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    alphas: &[f64],
) -> (f64, f64) {
    fit_once(pool, x, y, n, d, alphas, Device::Cpu);
    fit_once(pool, x, y, n, d, alphas, Device::Gpu);
    let (mut host, mut dev) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..reps() {
        let t0 = Instant::now();
        fit_once(pool, x, y, n, d, alphas, Device::Cpu);
        host = host.min(t0.elapsed().as_secs_f64() * 1e3);
        let t1 = Instant::now();
        fit_once(pool, x, y, n, d, alphas, Device::Gpu);
        dev = dev.min(t1.elapsed().as_secs_f64() * 1e3);
    }
    (host, dev)
}

/// `true` when this backend has a second arm to compare against; prints why not
/// otherwise, so a vacuous run cannot read as a passing one.
fn have_device(d: usize, what: &str) -> bool {
    if !gcv_device_possible(d) || !gcv_device_arm(Device::Gpu, GcvMode::Auto, 1000, d, 3, 1) {
        println!(
            "RIDGECV {what}: SKIPPED — no device GCV arm on backend={} at d={d}",
            capability::active_backend_name()
        );
        return false;
    }
    true
}

#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn device_vs_host_shape_ladder() {
    let alphas = logspace(-3.0, 3.0, 30);
    if !have_device(64, "shape ladder") {
        return;
    }
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    println!(
        "\nRIDGECV host-vs-device shape ladder  30 alphas  backend={}\n  \
         (upload INSIDE the device timer — the ingress a host-slice fit pays)",
        capability::active_backend_name()
    );
    let before = cpu_sample();
    for (n, d) in [
        (10_000usize, 16usize),
        (10_000, 64),
        (100_000, 16),
        (100_000, 64),
        (100_000, 128),
        (50_000, 256),
        (200_000, 64),
    ] {
        if !gcv_device_possible(d) {
            println!("  n={n:>7} d={d:>4}: no device arm at this d");
            continue;
        }
        let (x, y) = make_regression(n, d, 42);
        let (host, dev) = measure_pair(&mut pool, &x, &y, n, d, &alphas);
        println!(
            "  n={n:>7} d={d:>4}: host {host:8.2} ms   device {dev:8.2} ms   \
             {:5.2}x",
            host / dev
        );
    }
    report_foreign(before, "the shape ladder");
}

#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn device_vs_host_alphas_ladder() {
    let (n, d) = (100_000usize, 64usize);
    if !have_device(d, "alphas ladder") {
        return;
    }
    let (x, y) = make_regression(n, d, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    println!(
        "\nRIDGECV host-vs-device alphas ladder  n={n} d={d}  backend={}\n  \
         (the per-alpha term is what grows the arithmetic per uploaded element)",
        capability::active_backend_name()
    );
    let before = cpu_sample();
    for k in [1usize, 3, 10, 30, 100, 200] {
        let alphas = logspace(-3.0, 3.0, k);
        let (host, dev) = measure_pair(&mut pool, &x, &y, n, d, &alphas);
        println!(
            "  alphas={k:>4}: host {host:8.2} ms   device {dev:8.2} ms   {:5.2}x",
            host / dev
        );
    }
    report_foreign(before, "the alphas ladder");
}

/// Where a device fit's time goes: upload, normal equations, sweep.
///
/// Measured through the prim rather than the estimator so each phase is timed on
/// its own; the estimator's own total is printed alongside so the three parts
/// can be checked against the whole (the residual is the host `sym_eig` plus the
/// `O(n_alphas·d²·n_y)` coefficient block).
#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn device_phase_attribution() {
    let alphas = logspace(-3.0, 3.0, 30);
    if !have_device(64, "phase attribution") {
        return;
    }
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    println!(
        "\nRIDGECV device phase attribution  30 alphas  backend={}",
        capability::active_backend_name()
    );
    let before = cpu_sample();
    for (n, d) in [(100_000usize, 64usize), (100_000, 128), (50_000, 256)] {
        if !gcv_device_possible(d) {
            continue;
        }
        let (x, y) = make_regression(n, d, 42);
        let sqrt_sw = vec![1.0f64; n];

        let mut up = f64::INFINITY;
        let mut ne = f64::INFINITY;
        let mut sw = f64::INFINITY;
        for _ in 0..reps() {
            let t0 = Instant::now();
            let dev =
                GcvDevice::from_host::<f64>(&mut pool, &x, &y, n, d, 1, &sqrt_sw, false).unwrap();
            // The upload is asynchronous on some runtimes; the first blocking
            // read is what pays for it, so the phase below carries any tail.
            up = up.min(t0.elapsed().as_secs_f64() * 1e3);

            let t1 = Instant::now();
            let (xm, ym, gram, xty, xtsw) = dev.normal_equations(&mut pool, true).unwrap();
            ne = ne.min(t1.elapsed().as_secs_f64() * 1e3);

            // Operands of the right SHAPE; the sweep's cost does not depend on
            // their values, and re-deriving the eigendecomposition here would
            // time the host phase this probe is trying to exclude.
            let v = vec![0.01f64; d * d];
            let g = vec![0.5f64; alphas.len() * d];
            let gz = vec![0.1f64; alphas.len() * d];
            let gzsw = vec![0.01f64; alphas.len() * d];
            let _ = (&gram, &xty, &xtsw);
            let t2 = Instant::now();
            dev.sweep(
                &mut pool,
                &xm,
                &ym,
                &v,
                &g,
                &gz,
                &gzsw,
                alphas.len(),
                n as f64,
                true,
                false,
                false,
            )
            .unwrap();
            sw = sw.min(t2.elapsed().as_secs_f64() * 1e3);
            dev.release_into(&mut pool);
        }

        let total = {
            let mut best = f64::INFINITY;
            fit_once(&mut pool, &x, &y, n, d, &alphas, Device::Gpu);
            for _ in 0..reps() {
                let t = Instant::now();
                fit_once(&mut pool, &x, &y, n, d, &alphas, Device::Gpu);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let bytes = (n * d * 8) as f64;
        println!(
            "  n={n:>7} d={d:>4}: upload {up:7.2} ms ({:5.2} GB/s)   \
             normal_eq {ne:7.2} ms   sweep {sw:7.2} ms   whole fit {total:7.2} ms",
            bytes / (up * 1e6)
        );
    }
    report_foreign(before, "the phase attribution");
}
