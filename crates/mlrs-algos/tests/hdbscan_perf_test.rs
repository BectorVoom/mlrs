//! `Hdbscan` (HDBS-01) `fit` wall-clock performance probe + stage breakdown.
//!
//! A plain `std::time::Instant` probe (the `knn_classifier_fit_perf_test.rs`
//! precedent). `#[ignore]` by default; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test hdbscan_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_hdbscan.py` (sklearn) on the SAME splitmix64
//! blob ladder. `fit` is the whole pipeline: core distances → mutual-reachability
//! MST → single linkage → condensed tree → EoM selection. The GLOSH
//! `outlier_scores_` pass is NOT part of `fit` (HDBS-PERF-CPU deferred it to
//! first access); `hdbscan_fit_stage_breakdown` still times it so the deferred
//! cost stays visible.
//!
//! `HDBSCAN_PERF_METRIC` selects the metric (default euclidean);
//! `HDBSCAN_PERF_MAX_N` caps the ladder. As of HDBS-PARAMS the Python shim
//! exposes every metric too, so `scripts/bench_hdbscan_params.py` can sweep them
//! against sklearn end-to-end — this file remains the way to time a single
//! STAGE (core distances, MST) without the ingress/egress around it.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::cluster::hdbscan::{Hdbscan, Metric};
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_hdbscan.py`).
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

/// Well-separated blobs, matching `scripts/bench_hdbscan.py::make_blobs`.
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

fn max_n() -> usize {
    std::env::var("HDBSCAN_PERF_MAX_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

/// The metric under test (`HDBSCAN_PERF_METRIC`, default euclidean). The Python
/// shim only exposes euclidean today, so the other four are measurable only from
/// here.
fn perf_metric() -> Metric {
    match std::env::var("HDBSCAN_PERF_METRIC").as_deref() {
        Ok("manhattan") => Metric::Manhattan,
        Ok("chebyshev") => Metric::Chebyshev,
        Ok("minkowski") => Metric::Minkowski { p: 3.0 },
        Ok("cosine") => Metric::Cosine,
        _ => Metric::Euclidean,
    }
}

/// Best-of-`reps` full-`fit` seconds for one `(n, d, mcs, k)` config.
fn fit_seconds(n: usize, d: usize, mcs: usize, k: usize, reps: usize) -> f64 {
    let x_host = make_blobs(n, d, k, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let est = Hdbscan::<f64>::builder()
            .min_cluster_size(mcs)
            .metric(perf_metric())
            .build::<f64>()
            .expect("valid hyperparameters");
        let t0 = Instant::now();
        let fitted = Fit::fit(est, &mut pool, &x, None, (n, d)).expect("fit succeeds");
        let secs = t0.elapsed().as_secs_f64();
        // Touch the labels so nothing is optimized away.
        std::hint::black_box(fitted.labels(&pool).len());
        best = best.min(secs);
    }
    best
}

/// Stage-by-stage breakdown of the euclidean (Variant-B) fit pipeline, replayed
/// through the same public submodule entry points `Fit::fit` calls. Tells us
/// WHICH stage owns the wall clock before any optimization lands.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn hdbscan_fit_stage_breakdown() {
    let _ = env_logger::builder().is_test(true).try_init();
    use mlrs_algos::cluster::hdbscan::{
        condense, glosh, host_core, mst, select, single_linkage, stability,
    };
    use mlrs_backend::prims::knn_graph::{knn_graph, Metric as KnnMetric};

    let n: usize = std::env::var("HDBSCAN_STAGE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);
    let d = std::env::var("HDBSCAN_STAGE_D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);
    let mcs = std::env::var("HDBSCAN_STAGE_MCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5usize);
    // The device KNN prim is the pre-HDBS-PERF-CPU core-distance source and takes
    // ~4 minutes at n=1000 on the cpu backend; time it only when asked.
    let with_device = std::env::var("HDBSCAN_STAGE_DEVICE_KNN").is_ok();
    let x_host = make_blobs(n, d, 6, 42);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    macro_rules! stage {
        ($label:expr, $body:expr) => {{
            let t = Instant::now();
            let out = $body;
            println!("  {:<26} {:>10.4} s", $label, t.elapsed().as_secs_f64());
            out
        }};
    }

    println!("n={n} d={d} mcs={mcs}");
    let k = mcs.min(n);
    if with_device {
        let (idx_dev, dist_dev) = stage!("knn_graph (device, OLD)", {
            knn_graph::<f64>(&mut pool, &x, (n, d), k, KnnMetric::Euclidean, true, 2.0).unwrap()
        });
        idx_dev.release_into(&mut pool);
        dist_dev.release_into(&mut pool);
    }
    let core_raw = stage!("core_distances_host", {
        host_core::core_distances_host(&x_host, n, d, Metric::Euclidean, k)
    });

    let edges = stage!("mst (specialized)", {
        mst::mst_from_data_matrix_metric(&x_host, n, d, &core_raw, 1.0, Metric::Euclidean)
    });
    let _ = stage!("mst (generic, OLD)", {
        mst::mst_from_data_matrix(&core_raw, n, 1.0, |i, j| {
            let (a, b) = (&x_host[i * d..(i + 1) * d], &x_host[j * d..(j + 1) * d]);
            a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt()
        })
    });
    let sorted = stage!("argsort_by_weight", mst::argsort_by_weight(&edges));
    let hierarchy = stage!("make_single_linkage", single_linkage::make_single_linkage(&sorted, n));
    let condensed = stage!("condense_tree", condense::condense_tree(&hierarchy, mcs));
    let stab = stage!("compute_stability", stability::compute_stability(&condensed));
    let _ = stage!("get_clusters", {
        select::get_clusters(&condensed, &stab, select::SelectionMethod::Eom, false, 0.0, 0, n)
    });

    // The GLOSH side pipeline — DEFERRED out of `fit` by HDBS-PERF-CPU; timed here
    // so the cost that moved to first-`outlier_scores_`-access stays visible.
    let dense = stage!("glosh: dense n×n", {
        let mut dist = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let (a, b) = (&x_host[i * d..(i + 1) * d], &x_host[j * d..(j + 1) * d]);
                dist[i * n + j] = a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt();
            }
        }
        dist
    });
    let _ = stage!("glosh: tree+scores", {
        glosh::hdbscan_outlier_scores(&dense, n, mcs, mcs, host_core::ALL_UNITS)
    });
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn hdbscan_fit_wall_clock() {
    let _ = env_logger::builder().is_test(true).try_init();
    let configs: [(usize, usize, usize, usize); 5] = [
        (1_000, 8, 5, 6),
        (2_000, 8, 5, 6),
        (5_000, 16, 10, 8),
        (10_000, 16, 25, 8),
        (20_000, 16, 25, 8),
    ];
    println!("{:>7} {:>4} {:>5} | {:>10}", "n", "d", "mcs", "fit (s)");
    for (n, d, mcs, k) in configs {
        if n > max_n() {
            continue;
        }
        let secs = fit_seconds(n, d, mcs, k, 3);
        println!("{n:>7} {d:>4} {mcs:>5} | {secs:>10.4}");
    }
}

/// Core-distance stage only: brute scan vs KD-tree across a `d` (and `n`) ladder.
///
/// This is the probe that sets `kdtree::KD_MAX_DIMS` / `KD_MIN_ROWS`. The box
/// bound loses its pruning power as `d` grows, so the tree has a crossover past
/// which it is pure overhead — that crossover is MEASURED here, never assumed.
/// Both routes are asserted bit-identical as they are timed, so a "win" can never
/// be a win on different numbers.
///
/// ```text
/// HDBSCAN_CORE_N=10000 HDBSCAN_CORE_MCS=10 cargo test -p mlrs-algos --release \
///   --features cpu --test hdbscan_perf_test hdbscan_core_distance_sweep \
///   -- --ignored --nocapture
/// ```
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn hdbscan_core_distance_sweep() {
    use mlrs_algos::cluster::hdbscan::host_core;
    use mlrs_backend::abflag;

    let n: usize = std::env::var("HDBSCAN_CORE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let mcs: usize = std::env::var("HDBSCAN_CORE_MCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let dims: Vec<usize> = std::env::var("HDBSCAN_CORE_DIMS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16, 24, 32, 48, 64]);
    let reps: usize = std::env::var("HDBSCAN_CORE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    println!("n={n} mcs={mcs} metric={:?} (best of {reps})", perf_metric());
    // Three routes: `brute` forces the scan, `kdtree` forces the tree with the
    // route calibration OFF, and `adaptive` is what actually ships (tree built,
    // calibration free to abandon it). `adaptive` ÷ `brute` is the overhead the
    // calibration costs on data where the tree does not pay off.
    println!(
        "{:>4} | {:>10} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "d", "brute (s)", "kdtree (s)", "adapt (s)", "speedup", "adapt/br", "visited%"
    );
    // `HDBSCAN_CORE_UNIFORM=1` swaps the blob ladder for UNIFORM noise — the
    // adversarial case for a KD-tree, since with no cluster structure every box
    // overlaps the query ball and the prune has nothing to cut. The `d` gate must
    // hold on this data, not just on the favourable clustered case.
    let uniform = std::env::var("HDBSCAN_CORE_UNIFORM").is_ok();
    for d in dims {
        let x = if uniform {
            let mut st = 42u64;
            (0..n * d).map(|_| uniform01(&mut st) * 20.0).collect()
        } else {
            make_blobs(n, d, 8, 42)
        };
        let k = mcs.min(n);
        let mut best = [f64::INFINITY; 3];
        let mut out: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..reps {
            for slot in 0..3 {
                let _g = match slot {
                    0 => abflag::force("MLRS_HDBSCAN_CORE_KD", "0"),
                    1 => abflag::force("MLRS_HDBSCAN_CORE_KD", "1"),
                    _ => abflag::clear("MLRS_HDBSCAN_CORE_KD"),
                };
                let t0 = Instant::now();
                let core = host_core::core_distances_host(&x, n, d, perf_metric(), k);
                let secs = t0.elapsed().as_secs_f64();
                best[slot] = best[slot].min(secs);
                out[slot] = core;
            }
        }
        // A faster stage that computes different numbers is not a faster stage.
        for i in 0..n {
            for (slot, label) in [(1usize, "kd-tree"), (2, "adaptive")] {
                assert_eq!(
                    out[slot][i].to_bits(),
                    out[0][i].to_bits(),
                    "{label} disagrees with brute at n={n} d={d} row={i}"
                );
            }
        }
        // The visited fraction the route calibration keys on: what ONE query
        // evaluates, as a percentage of the `n` the brute scan always evaluates.
        // Measured with the calibration FORCED OFF so it reports the tree's real
        // pruning, not the post-fallback mixture.
        let visited_pct = {
            let _g = abflag::force("MLRS_HDBSCAN_CORE_KD", "1");
            host_core::kd_visited_fraction_probe(&x, n, d, perf_metric(), k) * 100.0
        };
        println!(
            "{d:>4} | {:>10.4} {:>10.4} {:>10.4} {:>8.2} {:>9.2} {:>9.2}",
            best[0],
            best[1],
            best[2],
            best[0] / best[1],
            best[2] / best[0],
            visited_pct
        );
    }
}

/// Variant-B Prim stage only, swept over worker count (best of `reps`).
///
/// The Prim barriers once per step, so its worker count is a real trade and NOT
/// simply "take every core" — see `mst::mst_units`. Single-shot numbers for this
/// stage move by 10%+ run to run (the spin barrier is sensitive to what else is
/// runnable), so the default must be set from a best-of-N sweep, never one sample.
/// Every count produces the IDENTICAL edge list, so this only ever trades wall
/// clock — asserted here rather than assumed.
///
/// ```text
/// HDBSCAN_MST_N=10000 HDBSCAN_MST_D=16 cargo test -p mlrs-algos --release \
///   --features cpu --test hdbscan_perf_test hdbscan_mst_units_sweep \
///   -- --ignored --nocapture
/// ```
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn hdbscan_mst_units_sweep() {
    use mlrs_algos::cluster::hdbscan::{host_core, mst};
    use mlrs_backend::abflag;

    let n: usize = std::env::var("HDBSCAN_MST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let d: usize = std::env::var("HDBSCAN_MST_D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let mcs: usize = std::env::var("HDBSCAN_MST_MCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let reps: usize = std::env::var("HDBSCAN_MST_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let units: Vec<usize> = std::env::var("HDBSCAN_MST_UNITS_LADDER")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 6, 8, 10, 12, 14, 16]);

    let x = make_blobs(n, d, 8, 42);
    let metric = perf_metric();
    let core = host_core::core_distances_host(&x, n, d, metric, mcs.min(n));
    let reference = mst::mst_from_data_matrix_metric(&x, n, d, &core, 1.0, metric);

    println!("n={n} d={d} mcs={mcs} metric={metric:?} (best of {reps})");
    println!("{:>7} | {:>10}", "units", "mst (s)");
    for u in units {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let _g = abflag::force("MLRS_HDBSCAN_MST_UNITS", &u.to_string());
            let t0 = Instant::now();
            let edges = mst::mst_from_data_matrix_metric(&x, n, d, &core, 1.0, metric);
            best = best.min(t0.elapsed().as_secs_f64());
            assert_eq!(
                edges.len(),
                reference.len(),
                "units={u} changed the edge count"
            );
            for (e, (g, w)) in edges.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    (g.0, g.1, g.2.to_bits()),
                    (w.0, w.1, w.2.to_bits()),
                    "units={u} changed MST edge {e}"
                );
            }
        }
        println!("{u:>7} | {best:>10.4}");
    }
}

// ---------------------------------------------------------------------------
// HDBS-PARAMS: the `leaf_size` sweep behind `kdtree::DEFAULT_LEAF_SIZE` and the
// sklearn-parity `Hdbscan::new` default of 40.
//
// A KD-tree leaf is scanned linearly, so `leaf_size` trades traversal
// bookkeeping (many small leaves → more box tests and `perm` indirection)
// against wasted distance work (few large leaves → pairs evaluated that a
// tighter box would have pruned). Both defaults in the tree — 32 in Rust for
// callers who set none, 40 on the estimator for sklearn parity — are justified
// only by where this curve is flat.
//
// Forced onto the tree route (`MLRS_HDBSCAN_CORE_KD=1`) so the knob is LIVE:
// on the brute route nothing reads `leaf_size` and the sweep would report a
// row of identical numbers that looks like "leaf_size is free".
//
//   cargo test -p mlrs-algos --release --features cpu \
//     --test hdbscan_perf_test -- --ignored --nocapture leaf_size
// ---------------------------------------------------------------------------

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn hdbscan_leaf_size_sweep() {
    use mlrs_algos::cluster::hdbscan::{host_core, Algorithm};

    let n: usize = std::env::var("HDBSCAN_LEAF_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let mcs: usize = std::env::var("HDBSCAN_LEAF_MCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let dims: Vec<usize> = std::env::var("HDBSCAN_LEAF_DIMS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16]);
    let leaves: Vec<usize> = std::env::var("HDBSCAN_LEAF_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16, 32, 40, 64, 128, 256]);
    let reps: usize = std::env::var("HDBSCAN_LEAF_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    println!(
        "leaf_size sweep: n={n} mcs={mcs} metric={:?} (best of {reps}, tree route FORCED)",
        perf_metric()
    );
    print!("{:>4} |", "d");
    for ls in &leaves {
        print!(" {:>9}", format!("ls={ls}"));
    }
    println!("  {:>8}", "best");

    for d in dims {
        let x = make_blobs(n, d, 8, 42);
        let k = mcs.min(n);
        let mut times = Vec::with_capacity(leaves.len());
        // The reference output: every leaf_size must reproduce it EXACTLY, or
        // the number below is timing a different computation and means nothing.
        let mut reference: Option<Vec<f64>> = None;
        for &leaf_size in &leaves {
            let mut best = f64::INFINITY;
            let mut out = Vec::new();
            for _ in 0..reps {
                let t0 = Instant::now();
                out = host_core::core_distances_host_with(
                    &x,
                    n,
                    d,
                    perf_metric(),
                    k,
                    host_core::ScanOpts {
                        algorithm: Algorithm::KdTree,
                        leaf_size,
                        units: host_core::ALL_UNITS,
                    },
                );
                best = best.min(t0.elapsed().as_secs_f64());
            }
            match &reference {
                None => reference = Some(out),
                Some(r) => assert_eq!(
                    &out, r,
                    "d={d} leaf_size={leaf_size}: core distances diverged — the \
                     timing below would be comparing different computations"
                ),
            }
            times.push(best);
        }
        let fastest = times
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| leaves[i])
            .unwrap_or(0);
        print!("{d:>4} |");
        for t in &times {
            print!(" {t:>9.4}");
        }
        let spread = times.iter().cloned().fold(f64::MIN, f64::max)
            / times.iter().cloned().fold(f64::MAX, f64::min).max(1e-12);
        println!("  ls={fastest:<5} spread={spread:.2}x");
    }
}
