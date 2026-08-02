//! `Ridge()` — the DEFAULT `positive=False` / `solver='cholesky'` arm —
//! wall-clock performance probe (RIDGE-DEFAULT-CUDA).
//!
//! The pre-existing `ridge_perf_test.rs` ladder pre-uploads the design OUTSIDE
//! the timer and stops at `d = 64`. Both were wrong for this campaign:
//!
//! - A Python caller passes a numpy array, so the upload is part of `fit` and
//!   has to be inside the timer. On a T4 it is 85–91% of a device-arm fit
//!   ([[mlrs-ridge-positive-cuda]]), which makes an upload-free ladder measure
//!   something no user experiences.
//! - `d > 64` was not slow, it was an ERROR — `cholesky_solve` rejected any
//!   order above the shared-memory kernel's `MAX_DIM`. That is also precisely
//!   the regime where a GPU fit can beat a CPU one, since the arithmetic grows
//!   as `n·d²/2` over an `n·d` transfer. The ladder here runs to `d = 256`.
//!
//! Like `ridge_positive_perf_test.rs`, this reproduces the PyO3
//! `ridge_fit_dispatch!` branch exactly — `host_fit_applicable`, then either the
//! no-upload host ingress or the device ingress with the upload timed — so a
//! `--features cpu` number and a `--features cuda` number measure the same
//! user-visible operation and can be divided.
//!
//! ```text
//! # the mlrs CPU baseline (the number the cuda arm has to beat)
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test ridge_default_perf_test -- --ignored --nocapture
//!
//! # the CUDA number (Colab T4)
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test ridge_default_perf_test -- --ignored --nocapture
//! ```
//!
//! `RIDGE_PROFILE=1` adds `ridge.rs`'s per-phase attribution. `MLRS_RIDGE_REPS`
//! overrides the min-of-N repeat count (default 5); the first config runs a
//! discarded warmup so pipeline compilation is excluded.
//!
//! `MLRS_RIDGE_GRAM_HOST=0`/`1` forces the device / host arm at any size, which
//! is how the two are compared ON ONE MACHINE — the second ladder this probe
//! prints does exactly that, because "the gpu beat the cpu" is only meaningful
//! against mlrs's own best host arm and not against whatever the default
//! dispatch happened to pick.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_ridge.py` and
/// `ridge_perf_test.rs`, so every engine fits the same dataset).
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

/// Same deterministic design as `ridge_perf_test.rs::make_regression`.
fn make_regression(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();

    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();

    let mut sn = seed + 2;
    let mut y = Vec::with_capacity(n);
    for r in 0..n {
        let mut dot = 0.5f64;
        for c in 0..d {
            dot += x[r * d + c] * coef[c];
        }
        dot += 0.01 * uniform_pm1(&mut sn);
        y.push(dot);
    }

    (
        x.iter().map(|&v| v as f32).collect(),
        y.iter().map(|&v| v as f32).collect(),
    )
}

/// One timed sample.
struct Sample {
    /// End-to-end seconds, upload included.
    total: f64,
    /// Seconds spent on ingress, and 0 unless `attribute` was set. Measuring it
    /// requires a blocking drain between the upload and the fit, which FORBIDS
    /// any overlap between them — so a sample with `ingress > 0` has a `total`
    /// that runs high and must not be used for the headline number.
    ingress: f64,
    /// `"host"` or `"device"` — which ingress `host_fit_applicable` chose.
    arm: &'static str,
}

/// Drain the queue so a lap ends where the caller thinks it does.
///
/// `client.sync()` returns a FUTURE — `let _ = pool.client().sync()` does
/// nothing and every lap silently bleeds into the next blocking read-back
/// (RIDGE-POS-PERF). A one-element `to_host` is a real blocking readback and is
/// the only reliable barrier available here.
fn drain(pool: &mut BufferPool<ActiveRuntime>, probe: &DeviceArray<ActiveRuntime, f32>) {
    let v = probe.to_host(pool);
    assert!(v[0].is_finite());
}

/// One DEFAULT (`positive=False`) fit from HOST data, through whichever ingress
/// `host_fit_applicable` selects — the PyO3 `ridge_fit_dispatch!` branch.
fn run_once(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    attribute: bool,
) -> Sample {
    let est = Ridge::<f32>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<f32>()
        .expect("build");

    let host_arm = est.host_fit_applicable((n, d));
    let t0 = Instant::now();
    let (intercept, ingress) = if host_arm {
        let fitted = est
            .fit_from_host_slice(pool, x, y, (n, d), None)
            .expect("host fit");
        (fitted.intercept(pool), 0.0)
    } else {
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(pool, x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(pool, y);
        let ingress = if attribute {
            drain(pool, &y_dev);
            t0.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let fitted = est
            .fit(pool, &x_dev, Some(&y_dev), (n, d))
            .expect("device fit");
        // The fitted state is device-resident (D-03); this one-element read is
        // what makes the timer include every queued kernel.
        let i = fitted.intercept(pool);
        x_dev.release_into(pool);
        y_dev.release_into(pool);
        (i, ingress)
    };
    let total = t0.elapsed().as_secs_f64();

    assert!(
        intercept.is_finite(),
        "degenerate fit at n={n} d={d} — perf run is broken"
    );
    Sample {
        total,
        ingress,
        arm: if host_arm { "host" } else { "device" },
    }
}

/// The ladder. `d = 128` and `d = 256` are the shapes the shared-memory Cholesky
/// cap used to reject outright, and the ones where a device fit has a chance.
const CONFIGS: &[(usize, usize)] = &[
    (1_000, 8),
    (10_000, 16),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (500_000, 16),
    (100_000, 128),
    (100_000, 256),
];

fn reps() -> usize {
    std::env::var("MLRS_RIDGE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Min-of-N over `reps()` samples plus one extra drained pass for the
/// upload/compute split.
fn measure(pool: &mut BufferPool<ActiveRuntime>, x: &[f32], y: &[f32], n: usize, d: usize) {
    let mut best = f64::INFINITY;
    let mut arm = "";
    for _ in 0..reps() {
        let s = run_once(pool, x, y, n, d, false);
        best = best.min(s.total);
        arm = s.arm;
    }
    let split = run_once(pool, x, y, n, d, true);
    println!(
        "{n:>9} {d:>5} | {:>10.4} {:>10.4} {:>10.4} | {arm:>6}",
        best * 1e3,
        split.ingress * 1e3,
        (split.total - split.ingress) * 1e3,
    );
}

/// The headline ladder: whatever `host_fit_applicable` picks, which is what a
/// Python caller gets.
fn headline_ladder(pool: &mut BufferPool<ActiveRuntime>) {
    println!(
        "backend={} reps={} solver=cholesky (positive=False)",
        mlrs_backend::capability::active_backend_name(),
        reps()
    );
    println!(
        "{:>9} {:>5} | {:>10} {:>10} {:>10} | {:>6}",
        "n", "d", "fit (ms)", "upload", "compute", "arm"
    );

    for (i, &(n, d)) in CONFIGS.iter().enumerate() {
        let (x, y) = make_regression(n, d, 42);
        if i == 0 {
            run_once(pool, &x, &y, n, d, false); // warmup (JIT + first-touch)
        }
        measure(pool, &x, &y, n, d);
    }
}

/// The A/B that answers "does the gpu beat the cpu": BOTH arms, on ONE machine,
/// at every rung.
///
/// Run under `--features cuda` this prints the device arm against mlrs's own
/// host arm on the same box, which is the honest comparison — the default
/// dispatch would otherwise silently take the host arm at the small rungs and
/// the device arm at the large ones, and the resulting column could not be
/// divided by anything.
///
/// The host arm here is the SAME code the `--features cpu` build runs; the only
/// difference is the machine's core count. On a Colab VM (2 vCPU) that host
/// number is roughly 4× the local 16-thread box's, which is why the local
/// baseline is the one to quote for a "beats the CPU" claim and the Colab
/// baseline is the one to quote for "on this machine".
fn arm_ab_ladder(pool: &mut BufferPool<ActiveRuntime>) {
    let backend = mlrs_backend::capability::active_backend_name();
    if backend == "cpu" {
        // Forcing the DEVICE arm on the cpu backend does not measure a slower
        // arm, it hangs: `gram_path` there is the `gemm` fallback, so
        // `gram_xty_centered` defers to `center_columns`, whose cpu arm walks
        // the `d` columns one at a time with an upload + launch + blocking
        // readback each. That composition took 59.6 s of a 60.1 s `1 000 × 8`
        // fit — and this ladder's largest rung is `100 000 × 256`. The host arm
        // exists precisely so nothing takes that route; there is no comparison
        // to print.
        println!("backend=cpu — forced-arm A/B SKIPPED (the cpu device arm is the ");
        println!("  `center_columns` per-column round-trip pathology the host arm replaces)");
        return;
    }
    println!("backend={backend} reps={} — forced-arm A/B", reps());
    println!(
        "{:>9} {:>5} | {:>11} {:>11} | {:>9}",
        "n", "d", "host (ms)", "device (ms)", "speedup"
    );

    for (i, &(n, d)) in CONFIGS.iter().enumerate() {
        let (x, y) = make_regression(n, d, 42);
        if i == 0 {
            let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
            run_once(pool, &x, &y, n, d, false);
        }

        let mut host = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
            for _ in 0..reps() {
                let s = run_once(pool, &x, &y, n, d, false);
                assert_eq!(s.arm, "host", "the =1 force must reach the dispatcher");
                host = host.min(s.total);
            }
        }
        let mut dev = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
            for _ in 0..reps() {
                let s = run_once(pool, &x, &y, n, d, false);
                assert_eq!(s.arm, "device", "the =0 force must reach the dispatcher");
                dev = dev.min(s.total);
            }
        }
        println!(
            "{n:>9} {d:>5} | {:>11.4} {:>11.4} | {:>8.2}x",
            host * 1e3,
            dev * 1e3,
            host / dev,
        );
    }
}

/// Both ladders, in ONE `#[test]`.
///
/// Deliberately not two test functions: `libtest` runs a binary's tests on
/// PARALLEL threads, and two wall-clock probes sharing one GPU interleave into
/// numbers that are neither's. (That is not hypothetical — the first run of this
/// file produced a headline `100 000 × 256` of 216.9 ms against the A/B's
/// 184.1 ms for the same fit, purely from contention.) One test, run in order,
/// removes the footgun rather than documenting a `--test-threads=1` the next
/// reader will forget.
#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn ridge_default_perf() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    headline_ladder(&mut pool);
    println!();
    arm_ab_ladder(&mut pool);
}
