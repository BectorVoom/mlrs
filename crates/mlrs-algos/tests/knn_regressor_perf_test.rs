//! `KNeighborsRegressor` (NEIGH-03) wall-clock performance probe.
//!
//! A plain `std::time::Instant` probe (the `ridge_perf_test.rs` precedent — NOT
//! a Criterion micro-benchmark). `#[ignore]` by default so the ordinary suite
//! stays fast; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test knn_regressor_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_knn.py` (sklearn, and cuML on a CUDA host) on
//! the SAME splitmix64 design matrix, so every engine fits the byte-identical
//! dataset.
//!
//! The second probe (`knn_stage_breakdown`) attributes the predict wall-clock to
//! the two device stages (`distance` GEMM-expansion vs `top_k` selection) so the
//! optimization target is measured, not guessed.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_knn.py`).
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

/// Deterministic random design + linear target (identical generator to the
/// Ridge/LinearRegression probes so the datasets are comparable).
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

/// `(fit_seconds, predict_seconds)` for one `(n_train, d, k, n_query)` config.
fn run_config(n: usize, d: usize, k: usize, n_query: usize) -> (f64, f64) {
    let (x, y) = make_regression(n, d, 42);
    let (xq, _) = make_regression(n_query, d, 7);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

    let t0 = Instant::now();
    let fitted = KNeighborsRegressor::<f32>::builder()
        .n_neighbors(k)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("fit");
    let fit_s = t0.elapsed().as_secs_f64();

    // BEST-OF-3 (KNN-02): a single cold call folds NVRTC JIT time and the
    // GPU's idle-clock ramp into the measurement — the ladder's early configs
    // were reporting boost-ramp wall-clock, not kernel throughput (the same
    // kernel measured 6.2 Gpair/s early in the ladder and 10.6 late). The
    // sklearn/cuML side (`scripts/bench_knn.py`) uses the same best-of-3.
    let mut predict_s = f64::INFINITY;
    for _ in 0..3 {
        let t1 = Instant::now();
        let pred = fitted
            .predict(&mut pool, &xq_dev, (n_query, d))
            .expect("predict");
        // Force completion of every queued kernel with the boundary readback.
        let host = pred.to_host(&mut pool);
        let elapsed = t1.elapsed().as_secs_f64();
        assert_eq!(host.len(), n_query);
        pred.release_into(&mut pool);
        if elapsed < predict_s {
            predict_s = elapsed;
        }
    }

    (fit_s, predict_s)
}

/// Skip configs whose `n_train` exceeds `KNN_PERF_MAX_N`, if set.
///
/// The full ladder is sized for a real datacentre GPU. On a slow backend (an
/// integrated adapter under wgpu, or the cpu runtime) the top configs take tens
/// of minutes, which is long enough to be mistaken for a hang — cap the ladder
/// there rather than trimming it for everyone.
fn max_n() -> usize {
    std::env::var("KNN_PERF_MAX_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_regressor_wall_clock() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Boost-clock warmup (KNN-03): the ladder runs small configs first, so on a
    // GPU that ramps clocks under load the EARLY configs report ramp wall-clock,
    // not kernel throughput — and the sklearn/cuML side of the comparison runs
    // in a separate later process against an already-hot GPU. One throwaway
    // predict at a large-ish shape levels the field; `KNN_PERF_NO_WARMUP=1`
    // restores the cold ladder.
    if std::env::var("KNN_PERF_NO_WARMUP").map(|v| v != "1").unwrap_or(true) {
        let n = 100_000.min(max_n());
        let (fit_s, pred_s) = run_config(n, 32, 10, 10_000);
        println!("# warmup (untimed): n={n} fit={fit_s:.4} predict={pred_s:.4}");
    }
    println!(
        "{:>8} {:>5} {:>4} {:>8} {:>12} {:>12}",
        "n_train", "d", "k", "n_query", "fit_s", "predict_s"
    );
    let cap = max_n();
    // `KNN_PERF_CONFIGS="n,d,k,nq;n,d,k,nq"` overrides the ladder — used for
    // shape probes (e.g. separating an n_query grid-size effect from an
    // n_train one) without a source edit per probe.
    let ladder: Vec<(usize, usize, usize, usize)> = match std::env::var("KNN_PERF_CONFIGS") {
        Ok(s) => s
            .split(';')
            .filter(|c| !c.trim().is_empty())
            .map(|c| {
                let v: Vec<usize> = c.split(',').map(|x| x.trim().parse().expect("config")).collect();
                (v[0], v[1], v[2], v[3])
            })
            .collect(),
        Err(_) => vec![
            (10_000, 16, 5, 2_000),
            (50_000, 16, 5, 10_000),
            (100_000, 32, 10, 10_000),
            (200_000, 32, 10, 20_000),
        ],
    };
    for &(n, d, k, nq) in &ladder {
        if n > cap {
            println!("{n:>8} {d:>5} {k:>4} {nq:>8} {:>12} {:>12}", "skipped", "skipped");
            continue;
        }
        let (fit_s, pred_s) = run_config(n, d, k, nq);
        println!("{n:>8} {d:>5} {k:>4} {nq:>8} {fit_s:>12.4} {pred_s:>12.4}");
    }
}

/// Attribute the fused-path predict wall-clock to its device stages DIRECTLY
/// (KNN-03 r6): `fused_distance_topk` (norms + fused kernel + optional merge +
/// sqrt) vs `knn_regress_mean_gather`, each timed with a sync, warm best-of-3,
/// over a (n_train, k) grid. This separates an in-kernel k-cost (fold/fill
/// phase) from a downstream one without needing nsys on the target image.
///
/// GPU-ONLY in practice: it calls `fused_distance_topk` DIRECTLY, and on the
/// cpu backend that is no longer the path `predict` dispatches (KNN-04 routes
/// cpu to the barrier-free `cpu_rows_topk`). Running it under `--features cpu`
/// measures the shared-memory kernels' spin-barrier behaviour, which takes
/// minutes per rung — use `knn_regressor_wall_clock` there instead.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_fused_stage_probe() {
    use mlrs_backend::prims::knn::{fused_distance_topk, knn_regress_mean_gather};

    let _ = env_logger::builder().is_test(true).try_init();
    let nq = 10_000usize;
    let d = 32usize;
    println!(
        "{:>8} {:>4} {:>12} {:>12}",
        "n_train", "k", "fused_s", "gather_s"
    );
    // Boost warmup at the largest shape.
    let cap = max_n().min(200_000);
    for &(n, k) in &[
        (cap, 10usize), // warmup row (discard mentally; also prints)
        (100_000, 1),
        (100_000, 4),
        (100_000, 10),
        (100_000, 16),
        (200_000, 1),
        (200_000, 10),
        (200_000, 16),
    ] {
        if n > max_n() {
            continue;
        }
        let (x, y) = make_regression(n, d, 42);
        let (xq, _) = make_regression(nq, d, 7);
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
        let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

        let mut fused_s = f64::INFINITY;
        let mut gather_s = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let (val, idx) =
                fused_distance_topk::<f32>(&mut pool, &xq_dev, (nq, d), &x_dev, (n, d), k, true)
                    .expect("fused");
            // Sync on a 1-element readback so every queued kernel completes.
            let probe: DeviceArray<ActiveRuntime, f32> =
                DeviceArray::from_raw(val.handle().clone(), 1);
            let _ = probe.to_host(&mut pool);
            let e0 = t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            let pred = knn_regress_mean_gather::<f32>(&mut pool, &y_dev, &idx, nq, k, n)
                .expect("gather");
            let host = pred.to_host(&mut pool);
            let e1 = t1.elapsed().as_secs_f64();
            assert_eq!(host.len(), nq);
            pred.release_into(&mut pool);
            val.release_into(&mut pool);
            idx.release_into(&mut pool);
            if e0 < fused_s {
                fused_s = e0;
            }
            if e1 < gather_s {
                gather_s = e1;
            }
        }
        println!("{n:>8} {k:>4} {fused_s:>12.4} {gather_s:>12.4}");
    }
}

/// Attribute the predict wall-clock to `distance` (GEMM-expansion) vs `top_k`
/// (per-row selection) so the hot stage is MEASURED rather than assumed, and A/B
/// the two selection kernels on THIS device.
///
/// The `MLRS_TOPK_SERIAL=1` arm is the legacy one-unit-per-row `select_k`; the
/// default arm is the block-parallel `select_k_shared`. Reporting both here is
/// what makes the dispatch decision a measurement on the target backend instead
/// of an extrapolation from a different one.
///
/// Sizes are kept modest because the SERIAL arm is the slow one — it is the
/// baseline being escaped from, and at larger `n_train` it runs long enough to
/// trip a GPU watchdog.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_stage_breakdown() {
    let _ = env_logger::builder().is_test(true).try_init();
    println!(
        "{:>8} {:>5} {:>4} {:>8} {:>12} {:>12} {:>12} {:>9}",
        "n_train", "d", "k", "n_query", "distance_s", "topk_par_s", "topk_ser_s", "speedup"
    );
    for &(n, d, k, nq) in &[
        (10_000usize, 16usize, 5usize, 2_000usize),
        (20_000, 16, 5, 4_000),
        (40_000, 32, 10, 4_000),
    ] {
        let (x, _) = make_regression(n, d, 42);
        let (xq, _) = make_regression(nq, d, 7);

        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

        let t0 = Instant::now();
        let dist = distance::<f32>(&mut pool, &xq_dev, (nq, d), &x_dev, (n, d), false, None)
            .expect("distance");
        // Sync on a 1-element readback so the GEMM is actually complete.
        let probe: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(dist.handle().clone(), 1);
        let _ = probe.to_host(&mut pool);
        let dist_s = t0.elapsed().as_secs_f64();

        std::env::remove_var("MLRS_TOPK_SERIAL");
        let t1 = Instant::now();
        let (val, idx) = top_k::<f32>(&mut pool, &dist, nq, n, k, true, None, None).expect("top_k");
        let _ = idx.to_host(&mut pool);
        let par_s = t1.elapsed().as_secs_f64();
        val.release_into(&mut pool);

        std::env::set_var("MLRS_TOPK_SERIAL", "1");
        let t2 = Instant::now();
        let (val, idx) = top_k::<f32>(&mut pool, &dist, nq, n, k, true, None, None).expect("top_k");
        let _ = idx.to_host(&mut pool);
        let ser_s = t2.elapsed().as_secs_f64();
        val.release_into(&mut pool);
        std::env::remove_var("MLRS_TOPK_SERIAL");

        let speedup = ser_s / par_s.max(1e-9);
        println!(
            "{n:>8} {d:>5} {k:>4} {nq:>8} {dist_s:>12.4} {par_s:>12.4} {ser_s:>12.4} {speedup:>8.1}x"
        );
    }
}
