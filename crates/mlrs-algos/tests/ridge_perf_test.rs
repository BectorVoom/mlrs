//! Ridge (LINEAR-02) wall-clock performance probe.
//!
//! A plain `std::time::Instant` probe (the `linear_regression_perf_test.rs`
//! precedent — NOT a Criterion micro-benchmark). `#[ignore]` by default so the
//! ordinary suite stays fast; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test ridge_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_ridge.py` (sklearn, and cuML on a CUDA
//! host) on the SAME splitmix64 design matrix, so every engine fits the
//! byte-identical dataset. Set `RIDGE_PROFILE=1` for per-phase
//! (`center`/`gram_xty`/`solve`) wall-clock attribution from `ridge.rs::fit`.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_ridge.py` and
/// the `linear_regression_perf_test.rs` precedent).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(state: &mut u64) -> f64 {
    // uniform in [-1, 1).
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Deterministic random design + linear target: `X` uniform in `[-1, 1)^d`
/// (seed `seed`), `true_coef` uniform in `[-1, 1)^d` (seed `seed+1`),
/// `y = X @ true_coef + 0.5 + 0.01 * noise` (noise stream seed `seed+2`). All
/// arithmetic in f64 then cast to f32, matching the Python generator exactly.
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

fn run_config(n: usize, d: usize) -> (f64, f64) {
    let (x, y) = make_regression(n, d, 42);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let t0 = Instant::now();
    let fitted = Ridge::<f32>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("fit");
    // fit's fitted state stays device-resident (D-03); force completion with a
    // tiny readback so the timing includes all queued kernels.
    let _ = fitted.intercept(&pool);
    let fit_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let pred = fitted
        .predict(&mut pool, &x_dev, (n, d))
        .expect("predict")
        .to_host(&pool);
    let predict_s = t1.elapsed().as_secs_f64();

    assert_eq!(pred.len(), n, "degenerate predict — perf run is broken");
    (fit_s, predict_s)
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn ridge_fit_predict_perf_ladder() {
    // Ridge has no direct-SVD/Gram+eig dual path (D-02: always Cholesky
    // normal-equations) and no feature cap, so this ladder targets the same
    // shapes as `linear_regression_perf_test.rs` for a direct comparison.
    let configs: &[(usize, usize)] = &[
        (200, 16),
        (10_000, 16),
        (10_000, 64),
        (100_000, 16),
        (100_000, 64),
        (500_000, 16),
        (1_000_000, 16),
    ];
    println!("{:>9} {:>4} | {:>10} {:>10}", "n", "d", "fit (s)", "pred (s)");
    // Warmup: first config twice so pipeline compilation is excluded from the
    // steady-state numbers (the first printed row still includes some warmup).
    for (i, &(n, d)) in configs.iter().enumerate() {
        if i == 0 {
            run_config(n, d);
        }
        let (fit_s, pred_s) = run_config(n, d);
        println!("{n:>9} {d:>4} | {fit_s:>10.4} {pred_s:>10.4}");
    }
}
