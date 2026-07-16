//! HistGradientBoosting (GBT-01) wall-clock performance probe.
//!
//! A plain `std::time::Instant` probe (the `random_forest_perf_test.rs`
//! precedent — NOT a Criterion micro-benchmark). `#[ignore]` by default so
//! the ordinary suite stays fast; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features wgpu \
//!   --test hist_gradient_boosting_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_hgb.py` (sklearn `HistGradientBoosting
//! Classifier`, this is its OpenMP home turf) on the SAME geometry and the
//! byte-identical splitmix64 dataset. The boosting loop is launch-only (one
//! host sync for the quantile edges), and the histogram gather is row-blocked
//! so shallow levels keep the device busy despite only `K` trees per launch.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier;
use mlrs_algos::typestate::{Fit, PredictLabels};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Deterministic host data: splitmix64-derived uniform features + a 3-class
/// rule with noise (byte-identical to `bench_hgb.py` / the RF probe).
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

fn make_data(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row_start = x.len();
        for _ in 0..d {
            x.push(uniform01(&mut s) as f32);
        }
        let a = x[row_start] as f64;
        let b = x[row_start + 1] as f64;
        let noise = uniform01(&mut s) < 0.05;
        let mut label = if a < 0.5 { 0 } else if b < 0.5 { 1 } else { 2 };
        if noise {
            label = (label + 1) % 3;
        }
        y.push(label as f32);
    }
    (x, y)
}

fn run_config(n: usize, d: usize, max_iter: usize, max_depth: usize) -> (f64, f64) {
    let (x, y) = make_data(n, d, 42);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let t0 = Instant::now();
    let clf = HistGradientBoostingClassifier::<f32>::builder()
        .max_iter(max_iter)
        .max_depth(max_depth)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("fit");
    // predict_labels forces a device sync (host readback), so the fit timing
    // below includes all queued fit kernels — time both phases together
    // first, then predict separately (already-synced queue).
    let labels = clf
        .predict_labels(&mut pool, &x_dev, (n, d))
        .expect("predict")
        .to_host(&pool);
    let fit_predict_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let labels2 = clf
        .predict_labels(&mut pool, &x_dev, (n, d))
        .expect("predict 2")
        .to_host(&pool);
    let predict_s = t1.elapsed().as_secs_f64();
    assert_eq!(labels, labels2, "predict must be deterministic");

    // Sanity: the ensemble must actually have learned the rule (not just
    // timed garbage) — train accuracy well above chance on 5%-noise data.
    let correct = labels
        .iter()
        .zip(y.iter())
        .filter(|&(&l, &t)| l == t as i32)
        .count();
    let acc = correct as f64 / n as f64;
    assert!(acc > 0.9, "train accuracy {acc} too low — perf run is broken");

    (fit_predict_s - predict_s, predict_s)
}

#[test]
#[ignore = "wall-clock perf probe — run targeted in release with --ignored --nocapture"]
fn hgb_fit_predict_wall_clock() {
    // (n, d, max_iter, depth) — the sklearn-comparison geometry ladder.
    let configs = [
        (10_000, 16, 100, 6),
        (50_000, 16, 100, 6),
        (100_000, 16, 100, 6),
        (50_000, 16, 200, 6),
        (50_000, 16, 100, 8),
    ];
    println!(
        "{:>8} {:>4} {:>6} {:>6} | {:>10} {:>10}",
        "n", "d", "iters", "depth", "fit (s)", "pred (s)"
    );
    for &(n, d, it, dep) in &configs {
        let (fit_s, pred_s) = run_config(n, d, it, dep);
        println!("{n:>8} {d:>4} {it:>6} {dep:>6} | {fit_s:>10.3} {pred_s:>10.3}");
    }
}
