//! KMeans (CLUSTER-01) wall-clock performance probe.
//!
//! A plain `std::time::Instant` probe (the `random_forest_perf_test.rs`
//! precedent — NOT a Criterion micro-benchmark). `#[ignore]` by default so the
//! ordinary suite stays fast; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features wgpu \
//!   --test kmeans_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_kmeans.py` (sklearn, and cuML on a CUDA
//! host) on the SAME splitmix64 blob data and the SAME injected init indices,
//! so every engine runs Lloyd from identical starting centers on the
//! byte-identical dataset. The fit loop is device-resident (per-iteration host
//! traffic is a few KB of sums/counts), so the number the probe prints is
//! dominated by kernel time, not synchronization.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::cluster::kmeans::KMeans;
use mlrs_algos::typestate::{Fit, PredictLabels};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_kmeans.py`).
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

/// Deterministic k-blob data: true centers uniform in `[0, 10)^d` (seed
/// `seed+1`), row `i` = `center[i % k] + uniform(-1, 1)` noise (seed `seed`).
/// All arithmetic in f64 then cast to f32, matching the numpy generator.
fn make_blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut cs = seed + 1;
    let centers: Vec<f64> = (0..k * d).map(|_| uniform01(&mut cs) * 10.0).collect();
    let mut s = seed;
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = i % k;
        for j in 0..d {
            x.push((centers[c * d + j] + (uniform01(&mut s) - 0.5) * 2.0) as f32);
        }
    }
    x
}

/// Deterministic k DISTINCT init row indices (seed `seed+2`, rejection on
/// duplicates — replicated exactly in the Python harness).
fn init_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut s = seed + 2;
    let mut idx: Vec<usize> = Vec::with_capacity(k);
    while idx.len() < k {
        let i = (splitmix64(&mut s) % n as u64) as usize;
        if !idx.contains(&i) {
            idx.push(i);
        }
    }
    idx
}

fn run_config(n: usize, d: usize, k: usize) -> (f64, f64, f64) {
    let x = make_blobs(n, d, k, 42);
    let init: Vec<f64> = init_indices(n, k, 42)
        .iter()
        .flat_map(|&i| x[i * d..(i + 1) * d].iter().map(|&v| v as f64))
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    let t0 = Instant::now();
    let fitted = KMeans::<f32>::builder()
        .n_clusters(k)
        .max_iter(300)
        .tol(1e-4)
        .init(Some(init))
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (n, d))
        .expect("fit");
    // fit ends with the labels_ boundary readback, so the timing includes all
    // queued kernels.
    let fit_s = t0.elapsed().as_secs_f64();
    let inertia = fitted.inertia() as f64;

    let t1 = Instant::now();
    let labels = fitted
        .predict_labels(&mut pool, &x_dev, (n, d))
        .expect("predict")
        .to_host(&pool);
    let predict_s = t1.elapsed().as_secs_f64();

    // Sanity: with well-separated blobs and one init row per true blob region,
    // the fit must actually use all k clusters.
    let mut seen = vec![false; k];
    for &l in labels.iter() {
        seen[l as usize] = true;
    }
    assert!(
        seen.iter().filter(|&&s| s).count() == k,
        "degenerate fit — perf run is broken"
    );

    (fit_s, predict_s, inertia)
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn kmeans_fit_predict_perf_ladder() {
    let configs: &[(usize, usize, usize)] = &[
        (100_000, 16, 8),
        (100_000, 64, 32),
        (500_000, 16, 8),
        (500_000, 32, 32),
        (1_000_000, 16, 8),
    ];
    println!(
        "{:>9} {:>4} {:>4} | {:>10} {:>10} {:>14}",
        "n", "d", "k", "fit (s)", "pred (s)", "inertia"
    );
    // Warmup: first config twice so pipeline compilation is excluded from the
    // steady-state numbers (the first printed row still includes some warmup).
    for (i, &(n, d, k)) in configs.iter().enumerate() {
        if i == 0 {
            run_config(n, d, k);
        }
        let (fit_s, pred_s, inertia) = run_config(n, d, k);
        println!("{n:>9} {d:>4} {k:>4} | {fit_s:>10.4} {pred_s:>10.4} {inertia:>14.6e}");
    }
}
