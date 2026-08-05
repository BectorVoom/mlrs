//! `Tsne` fit wall-clock probe + stage breakdown (TSNE-PARAMS).
//!
//! A plain `std::time::Instant` probe (the `umap_perf_test.rs` /
//! `hdbscan_perf_test.rs` precedent). `#[ignore]` by default; run TARGETED in
//! release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test tsne_perf_test -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is not optional. Every probe here saturates all cores, so
//! cargo's default of running them CONCURRENTLY has them measure each other:
//! observed, the first ladder rung came out 4.5x slow that way (0.78 s against
//! the 0.17 s a quiet machine gives). A ladder whose first rung is contaminated
//! reads as a scaling result, which is exactly the wrong conclusion to draw.
//!
//! `scripts/bench_tsne_params.py` is the sklearn-comparing half of this; it
//! sweeps the same parameters through the Python surface. This file exists for
//! what that one cannot see: WHERE inside a fit the time goes, and how the two
//! `method` arms and the `angle` dial scale on their own.
//!
//! ## Which parameters are perf-significant
//! `method` (the asymptotic class), `angle` (how much quadtree the negative
//! force walks), `perplexity` (which sets `n_neighbors = int(3·p + 1)`, sizing
//! both the graph and the edge loop), `n_components` (2 = quadtree, 3 =
//! octree), and `max_iter`. `init` / `verbose` / `min_grad_norm` /
//! `n_iter_without_progress` are not: the first two are free and the rest
//! change only when the descent STOPS, so timing them measures the stopping
//! rule rather than the implementation.
//!
//! `TSNE_PERF_N` / `TSNE_PERF_D` size the ladder.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::manifold::tsne::{Tsne, TsneInit, TsneMethod};
use mlrs_algos::manifold::tsne_knn::{bh_n_neighbors, joint_probabilities_nn, knn_graph};
use mlrs_algos::manifold::tsne_metric::{
    pairwise_squared, resolve_metric_params, MetricParams, TsneMetric,
};
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (the `umap_perf_test.rs` generator).
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

fn make_blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut sc = seed + 1;
    let centers: Vec<f64> = (0..k * d).map(|_| uniform01(&mut sc) * 30.0 - 15.0).collect();
    let mut sn = seed;
    let mut x = vec![0.0f64; n * d];
    for r in 0..n {
        let c = r % k;
        for j in 0..d {
            x[r * d + j] = centers[c * d + j] + (uniform01(&mut sn) - 0.5) * 1.4;
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

/// Best-of-`reps` full-`fit` seconds for one configuration.
#[allow(clippy::too_many_arguments)]
fn fit_seconds(
    x_host: &[f64],
    n: usize,
    d: usize,
    method: TsneMethod,
    angle: f64,
    perplexity: f64,
    n_components: usize,
    max_iter: usize,
    reps: usize,
) -> f64 {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, x_host);

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let est = Tsne::<f64>::builder()
            .method(method)
            .angle(angle)
            .perplexity(perplexity)
            .n_components(n_components)
            .max_iter(max_iter)
            .init(TsneInit::Pca)
            .seed(42)
            .build::<f64>()
            .expect("valid hyperparameters");
        let t0 = Instant::now();
        let fitted = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit succeeds");
        let secs = t0.elapsed().as_secs_f64();
        std::hint::black_box(fitted.kl_divergence());
        best = best.min(secs);
    }
    best
}

/// Stage-by-stage breakdown: which part of a Barnes-Hut fit owns the clock.
///
/// The setup stages are timed directly; the descent is inferred as the
/// remainder of a full fit, which is the only honest way to attribute it
/// without instrumenting the hot loop and changing what is being measured.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_fit_stage_breakdown() {
    let n = env_usize("TSNE_PERF_N", 3_000);
    let d = env_usize("TSNE_PERF_D", 8);
    let perplexity = 30.0f64;
    let x_host = make_blobs(n, d, 5, 42);
    let threads = mlrs_backend::capability::cpu_launch_units().max(1) as usize;

    println!("n={n} d={d} perplexity={perplexity} threads={threads}");

    let rp = resolve_metric_params(&x_host, n, d, TsneMetric::Euclidean, &MetricParams::default())
        .expect("resolve");

    let k = bh_n_neighbors(n, perplexity);
    let t0 = Instant::now();
    let graph = knn_graph(&x_host, n, d, k, TsneMetric::Euclidean, &rp, threads).expect("knn");
    let t_knn = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let sp = joint_probabilities_nn(&graph, n, perplexity, threads);
    let t_p = t0.elapsed().as_secs_f64();

    let total = fit_seconds(
        &x_host,
        n,
        d,
        TsneMethod::BarnesHut,
        0.5,
        perplexity,
        2,
        1000,
        1,
    );

    println!("  k (= int(3*perplexity+1), capped) = {k}, sparse P nnz = {}", sp.nnz());
    println!("  knn_graph            {t_knn:9.4} s");
    println!("  joint_probabilities  {t_p:9.4} s");
    println!("  descent (remainder)  {:9.4} s", total - t_knn - t_p);
    println!("  FULL FIT             {total:9.4} s");

    // The exact arm's setup, for contrast: a dense n×n pairwise matrix.
    if n <= 4_000 {
        let t0 = Instant::now();
        let dsq = pairwise_squared(&x_host, n, d, TsneMetric::Euclidean, &rp, threads)
            .expect("pairwise");
        println!(
            "  [exact] dense pairwise {:9.4} s ({} entries)",
            t0.elapsed().as_secs_f64(),
            dsq.len()
        );
    }
}

/// `method` — the asymptotic class. Barnes-Hut should pull away from exact as
/// `n` grows; that crossover is the whole reason the parameter exists.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_method_ladder() {
    let d = env_usize("TSNE_PERF_D", 8);
    println!("method ladder (d={d}, perplexity=30, max_iter=1000)");
    for n in [500usize, 1000, 2000] {
        let x = make_blobs(n, d, 5, 42);
        let bh = fit_seconds(&x, n, d, TsneMethod::BarnesHut, 0.5, 30.0, 2, 1000, 2);
        let ex = fit_seconds(&x, n, d, TsneMethod::Exact, 0.5, 30.0, 2, 1000, 2);
        println!("  n={n:5}  barnes_hut {bh:8.4} s   exact {ex:8.4} s   ({:5.2}x)", ex / bh);
    }
}

/// `angle` — the quadtree summary threshold. Lower θ visits more cells, so the
/// curve here IS the traversal cost, isolated from everything else.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_angle_ladder() {
    let n = env_usize("TSNE_PERF_N", 3_000);
    let d = env_usize("TSNE_PERF_D", 8);
    let x = make_blobs(n, d, 5, 42);
    println!("angle ladder (n={n}, d={d}, perplexity=30)");
    for angle in [0.2f64, 0.5, 0.8, 1.0] {
        let t = fit_seconds(&x, n, d, TsneMethod::BarnesHut, angle, 30.0, 2, 1000, 2);
        println!("  angle={angle:4.1}  {t:8.4} s");
    }
}

/// `perplexity` — sizes `n_neighbors = int(3·perplexity + 1)`, hence both the
/// one-off graph and the per-iteration edge loop.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_perplexity_ladder() {
    let n = env_usize("TSNE_PERF_N", 3_000);
    let d = env_usize("TSNE_PERF_D", 8);
    let x = make_blobs(n, d, 5, 42);
    println!("perplexity ladder (n={n}, d={d})");
    for perp in [5.0f64, 15.0, 30.0, 50.0] {
        let t = fit_seconds(&x, n, d, TsneMethod::BarnesHut, 0.5, perp, 2, 1000, 2);
        println!(
            "  perplexity={perp:5.1} (k={:3})  {t:8.4} s",
            bh_n_neighbors(n, perp)
        );
    }
}

/// `n_components` — 2 builds a quadtree (4 children per cell), 3 an octree (8),
/// with a correspondingly deeper and wider walk.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_n_components_ladder() {
    let n = env_usize("TSNE_PERF_N", 2_000);
    let d = env_usize("TSNE_PERF_D", 8);
    let x = make_blobs(n, d, 5, 42);
    println!("n_components ladder (n={n}, d={d}, perplexity=30)");
    for nc in [2usize, 3] {
        let t = fit_seconds(&x, n, d, TsneMethod::BarnesHut, 0.5, 30.0, nc, 1000, 2);
        println!("  n_components={nc}  {t:8.4} s");
    }
}

/// `n_jobs` — the worker count. Value-neutral by construction (every reduction
/// runs in point order, gated by `tsne_params_test::n_jobs_is_value_neutral`),
/// so this measures scaling only.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn tsne_n_jobs_scaling() {
    let n = env_usize("TSNE_PERF_N", 3_000);
    let d = env_usize("TSNE_PERF_D", 8);
    let x_host = make_blobs(n, d, 5, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    println!("n_jobs scaling (n={n}, d={d}, perplexity=30, barnes_hut)");
    let mut serial = f64::NAN;
    for n_jobs in [1i32, 2, 4, 8, 16] {
        let est = Tsne::<f64>::builder()
            .perplexity(30.0)
            .init(TsneInit::Pca)
            .n_jobs(Some(n_jobs))
            .seed(42)
            .build::<f64>()
            .expect("valid hyperparameters");
        let t0 = Instant::now();
        let fitted = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit succeeds");
        let secs = t0.elapsed().as_secs_f64();
        std::hint::black_box(fitted.kl_divergence());
        if n_jobs == 1 {
            serial = secs;
        }
        println!("  n_jobs={n_jobs:3}  {secs:8.4} s  ({:5.2}x over serial)", serial / secs);
    }
}
