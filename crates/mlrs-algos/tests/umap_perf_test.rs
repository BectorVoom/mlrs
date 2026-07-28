//! `Umap` (UMAP-01) `fit` wall-clock performance probe + stage breakdown.
//!
//! A plain `std::time::Instant` probe (the `hdbscan_perf_test.rs` precedent).
//! `#[ignore]` by default; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test umap_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_umap.py` (umap-learn) on the SAME splitmix64
//! blob ladder. `fit` is the whole pipeline: kNN graph → smooth-kNN ρ/σ →
//! membership → t-conorm union → a/b LM fit → init → SGD layout.
//!
//! `UMAP_PERF_MAX_N` caps the ladder; `UMAP_STAGE_N` / `UMAP_STAGE_D` /
//! `UMAP_STAGE_EPOCHS` size the breakdown.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::manifold::umap::{Init, Metric, Umap};
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_umap.py`).
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

/// Well-separated blobs, matching `scripts/bench_umap.py::make_blobs`.
fn make_blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut sc = seed + 1;
    let centers: Vec<f64> = (0..k * d).map(|_| uniform01(&mut sc) * 20.0).collect();
    let mut sn = seed;
    let mut x = vec![0.0f64; n * d];
    for r in 0..n {
        let c = r % k;
        for j in 0..d {
            x[r * d + j] = centers[c * d + j] + (uniform01(&mut sn) - 0.5) * 2.0;
        }
    }
    x
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn perf_metric() -> Metric {
    match std::env::var("UMAP_PERF_METRIC").as_deref() {
        Ok("manhattan") => Metric::Manhattan,
        Ok("chebyshev") => Metric::Chebyshev,
        Ok("minkowski") => Metric::Minkowski { p: 3.0 },
        Ok("cosine") => Metric::Cosine,
        _ => Metric::Euclidean,
    }
}

/// Best-of-`reps` full-`fit` seconds for one `(n, d, n_neighbors, epochs)` config.
fn fit_seconds(n: usize, d: usize, nn: usize, epochs: Option<usize>, reps: usize) -> f64 {
    let x_host = make_blobs(n, d, 6, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let est = Umap::<f64>::builder()
            .n_neighbors(nn)
            .n_components(2)
            .metric(perf_metric())
            .n_epochs(epochs)
            .init(Init::Spectral)
            .random_state(Some(42))
            .build::<f64>()
            .expect("valid hyperparameters");
        let t0 = Instant::now();
        let fitted = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit succeeds");
        let secs = t0.elapsed().as_secs_f64();
        std::hint::black_box(fitted.embedding(&pool).len());
        best = best.min(secs);
    }
    best
}

/// Stage-by-stage breakdown of the fit pipeline. Tells us WHICH stage owns the
/// wall clock before any optimization lands.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn umap_fit_stage_breakdown() {
    let _ = env_logger::builder().is_test(true).try_init();

    let n = env_usize("UMAP_STAGE_N", 1_000);
    let d = env_usize("UMAP_STAGE_D", 8);
    let nn = env_usize("UMAP_STAGE_NN", 15);
    let epochs = env_usize("UMAP_STAGE_EPOCHS", 500);

    let x_host = make_blobs(n, d, 6, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    println!("n={n} d={d} n_neighbors={nn} epochs={epochs}");

    // 1 epoch — isolates everything BEFORE the SGD layout (plus one epoch of it).
    let one = {
        let est = Umap::<f64>::builder()
            .n_neighbors(nn)
            .n_epochs(Some(1))
            .random_state(Some(42))
            .build::<f64>()
            .unwrap();
        let t0 = Instant::now();
        let f = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit");
        let s = t0.elapsed().as_secs_f64();
        std::hint::black_box(f.embedding(&pool).len());
        s
    };
    println!("  {:<28} {:>10.4} s", "fit @ 1 epoch (graph+init)", one);

    let full = {
        let est = Umap::<f64>::builder()
            .n_neighbors(nn)
            .n_epochs(Some(epochs))
            .random_state(Some(42))
            .build::<f64>()
            .unwrap();
        let t0 = Instant::now();
        let f = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit");
        let s = t0.elapsed().as_secs_f64();
        std::hint::black_box(f.embedding(&pool).len());
        s
    };
    println!("  {:<28} {:>10.4} s", format!("fit @ {epochs} epochs"), full);
    println!(
        "  {:<28} {:>10.4} s  ({:.4} s/epoch)",
        "=> layout SGD (derived)",
        full - one,
        (full - one) / (epochs.saturating_sub(1)).max(1) as f64
    );
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn umap_fit_wall_clock() {
    let _ = env_logger::builder().is_test(true).try_init();
    // (n, d, n_neighbors)
    let configs: [(usize, usize, usize); 4] = [
        (1_000, 8, 15),
        (2_000, 8, 15),
        (5_000, 16, 15),
        (10_000, 16, 30),
    ];
    let max_n = env_usize("UMAP_PERF_MAX_N", usize::MAX);
    println!("{:>8} {:>5} {:>5} {:>12}", "n", "d", "nn", "fit_s");
    for (n, d, nn) in configs {
        if n > max_n {
            continue;
        }
        let s = fit_seconds(n, d, nn, None, 1);
        println!("{n:>8} {d:>5} {nn:>5} {s:>12.4}");
    }
}
