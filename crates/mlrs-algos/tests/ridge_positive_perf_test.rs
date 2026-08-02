//! `Ridge(positive=True)` wall-clock performance probe (RIDGE-POS-CUDA).
//!
//! The `ridge_perf_test.rs` probe times the DEFAULT (Cholesky) solver and
//! pre-uploads the design outside the timer, which is the wrong measurement for
//! the `positive` arm: that arm has TWO ingress routes
//! (`Ridge::fit_from_host_slice` and `Fit::fit`), the PyO3 layer picks between
//! them with `Ridge::host_fit_applicable`, and the device route's cost is
//! dominated by the design upload the other route never pays. This probe
//! therefore reproduces `ridge_fit_dispatch!` exactly — same branch, upload
//! INSIDE the timer — so a cpu-backend number and a cuda-backend number measure
//! the same user-visible operation and can be divided.
//!
//! ```text
//! # the local 16-thread CPU baseline
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test ridge_positive_perf_test -- --ignored --nocapture
//!
//! # the CUDA number (Colab T4)
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test ridge_positive_perf_test -- --ignored --nocapture
//! ```
//!
//! `RIDGE_PROFILE=1` adds `ridge.rs`'s per-phase attribution. `MLRS_POS_REPS`
//! overrides the min-of-N repeat count (default 5); the first config runs a
//! discarded warmup so pipeline compilation is excluded.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_ridge.py` and
/// `ridge_perf_test.rs`).
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

/// One timed sample: `(total, ingress, compute, arm)` seconds.
struct Sample {
    total: f64,
    ingress: f64,
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

/// One `positive=True` fit from HOST data, through whichever ingress
/// `host_fit_applicable` selects — the PyO3 `ridge_fit_dispatch!` branch.
///
/// The device arm's upload is inside the timer because a Python caller pays it,
/// and is ALSO reported separately: on a discrete GPU this operation moves
/// `n·d` elements across PCIe to do `n·d²/2` multiply-adds, so the split
/// between "bytes in flight" and "arithmetic" is the whole design question.
/// The fit is drained with the `intercept` read-back the same way
/// `ridge_perf_test.rs` does, so queued kernels are not left uncounted.
/// `attribute` inserts a blocking drain between the upload and the fit so the
/// two can be told apart. That drain is itself a cost the real path does not
/// pay (it forbids any upload/launch overlap), so the reported `total` is only
/// trustworthy when `attribute` is false — the caller times the ladder without
/// it and takes the split from one extra pass.
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
        .positive(true)
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

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn ridge_positive_fit_perf_ladder() {
    let reps: usize = std::env::var("MLRS_POS_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    // The RIDGE-POS-PERF ladder, so these numbers line up with the cpu-campaign
    // table rather than needing a fresh baseline.
    let configs: &[(usize, usize)] = &[
        (1_000, 8),
        (10_000, 16),
        (10_000, 64),
        (100_000, 16),
        (100_000, 64),
        (500_000, 16),
        (100_000, 256),
    ];

    println!(
        "backend={} reps={reps}",
        mlrs_backend::capability::active_backend_name()
    );
    println!(
        "{:>9} {:>5} | {:>10} {:>10} {:>10} | {:>6}",
        "n", "d", "fit (ms)", "upload", "compute", "arm"
    );

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for (i, &(n, d)) in configs.iter().enumerate() {
        let (x, y) = make_regression(n, d, 42);
        if i == 0 {
            // Warmup: pipeline compilation / first-touch allocation.
            run_once(&mut pool, &x, &y, n, d, false);
        }
        let mut best = f64::INFINITY;
        let mut arm = "";
        for _ in 0..reps {
            let s = run_once(&mut pool, &x, &y, n, d, false);
            best = best.min(s.total);
            arm = s.arm;
        }
        // One extra drained pass purely for the upload/compute split.
        let split = run_once(&mut pool, &x, &y, n, d, true);
        println!(
            "{n:>9} {d:>5} | {:>10.4} {:>10.4} {:>10.4} | {arm:>6}",
            best * 1e3,
            split.ingress * 1e3,
            (split.total - split.ingress) * 1e3,
        );
    }
}
