//! `Arima`/`AutoArima` `fit` wall-clock performance probe (TSA-01).
//!
//! A plain `std::time::Instant` probe (the `umap_perf_test.rs` precedent).
//! `#[ignore]` by default; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test arima_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_arima.py` (statsmodels) on the same series.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::timeseries::{Arima, AutoArima};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

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

/// A stationary-ish AR(2)/MA(1)-flavored series (same shape as the oracle
/// fixture, just synthetic and length-scalable).
fn make_series(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut y = vec![0.0f64; n];
    let mut e_prev = 0.0f64;
    for t in 0..n {
        let e = (uniform01(&mut s) - 0.5) * 2.0;
        let prev1 = if t >= 1 { y[t - 1] } else { 0.0 };
        let prev2 = if t >= 2 { y[t - 2] } else { 0.0 };
        y[t] = 0.5 * prev1 - 0.2 * prev2 + e + 0.3 * e_prev;
        e_prev = e;
    }
    y
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn arima_fit_wall_clock() {
    let n = env_usize("ARIMA_PERF_N", 120);
    let orders: [(usize, usize, usize); 4] = [(1, 0, 0), (2, 0, 1), (3, 0, 2), (5, 0, 5)];

    let y_host = make_series(n, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let y: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    println!("n={n}");
    println!("{:>10} {:>12} {:>10}", "order", "fit_s", "converged");
    for (p, d, q) in orders {
        let est = Arima::<f64>::builder().order(p, d, q).build::<f64>().expect("valid order");
        let t0 = Instant::now();
        let fitted = est.fit(&pool, &y, n).expect("fit succeeds");
        let secs = t0.elapsed().as_secs_f64();
        std::hint::black_box(fitted.loglik());
        println!("{:>10} {:>12.6} {:>10}", format!("({p},{d},{q})"), secs, fitted.converged());
    }
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn auto_arima_fit_wall_clock() {
    let n = env_usize("ARIMA_PERF_N", 120);
    let max_p = env_usize("ARIMA_PERF_MAX_P", 3);
    let max_q = env_usize("ARIMA_PERF_MAX_Q", 3);

    let y_host = make_series(n, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let y: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    println!("n={n} max_p={max_p} max_q={max_q}");
    let t0 = Instant::now();
    let best = AutoArima::search::<f64>(&pool, &y, n, 0, max_p, max_q).expect("search converges");
    let secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(best.loglik());
    println!("auto_arima fit_s={secs:.6} order={:?}", best.order());
}
