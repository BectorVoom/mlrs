//! `KNeighborsClassifier` (NEIGH-02) `fit` wall-clock performance probe.
//!
//! A plain `std::time::Instant` probe (the `knn_regressor_perf_test.rs`
//! precedent). `#[ignore]` by default; run TARGETED in release mode:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test knn_classifier_fit_perf_test -- --ignored --nocapture
//! ```
//!
//! Compare against `scripts/bench_knn_classifier.py` (sklearn) on the SAME
//! splitmix64 design matrix. `fit` is the target here (not `predict`, which
//! KNN-03/KNN-04 already optimized): for k-NN there is no model to solve for,
//! so `fit` only stores the training set — but the mlrs core still validates
//! labels host-side and takes device-resident ownership of `x`, and THAT path
//! is what this probe measures and stage-breaks down.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::neighbors::classifier::KNeighborsClassifier;
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `scripts/bench_knn.py` /
/// `knn_regressor_perf_test.rs`).
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

/// Deterministic random design + a 3-class integer target (dot-product sign
/// buckets), so `classes_` collection has real work to do.
fn make_classification(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();

    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();

    let mut y = Vec::with_capacity(n);
    for r in 0..n {
        let mut dot = 0.0f64;
        for c in 0..d {
            dot += x[r * d + c] * coef[c];
        }
        let class = if dot < -0.15 {
            0.0
        } else if dot > 0.15 {
            2.0
        } else {
            1.0
        };
        y.push(class);
    }

    (
        x.iter().map(|&v| v as f32).collect(),
        y.iter().map(|&v| v as f32).collect(),
    )
}

fn max_n() -> usize {
    std::env::var("KNN_PERF_MAX_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

/// `fit_seconds` for one `(n_train, d, k)` config, BEST-OF-3 (KNN-02
/// methodology — a single cold call folds in JIT/allocator warmup).
fn run_config(n: usize, d: usize, k: usize) -> f64 {
    let (x, y) = make_classification(n, d, 42);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

        let t0 = Instant::now();
        let fitted = KNeighborsClassifier::<f32>::builder()
            .n_neighbors(k)
            .build::<f32>()
            .expect("build")
            .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
            .expect("fit");
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best {
            best = elapsed;
        }
        // `fitted` owns a device_copy of x. There is no public API to pull
        // that buffer back out and return it to the pool's free-list, so
        // `drop` here just deallocates it directly (NOT a free-list release)
        // — each of the 3 iterations pays a fresh allocation inside `fit`,
        // matching what the sklearn side of this comparison does too (a new
        // `SkKNN(...).fit(x, y)` per repeat, `scripts/bench_knn_classifier.py`),
        // so the engine-to-engine comparison stays fair.
        drop(fitted);
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
    }
    best
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_classifier_fit_wall_clock() {
    let _ = env_logger::builder().is_test(true).try_init();
    let cap = max_n();

    // Warmup at a mid-size shape so the timed ladder isn't paying one-time
    // allocator/JIT costs on its first real row.
    let warm_n = 20_000.min(cap);
    let warm_s = run_config(warm_n, 32, 10);
    println!("# warmup (untimed): n={warm_n} fit={warm_s:.6}");

    println!("{:>8} {:>5} {:>4} {:>12}", "n_train", "d", "k", "fit_s");
    let ladder: Vec<(usize, usize, usize)> = match std::env::var("KNN_PERF_CONFIGS") {
        Ok(s) => s
            .split(';')
            .filter(|c| !c.trim().is_empty())
            .map(|c| {
                let v: Vec<usize> = c.split(',').map(|x| x.trim().parse().expect("config")).collect();
                assert_eq!(
                    v.len(),
                    3,
                    "KNN_PERF_CONFIGS group {c:?} must be `n,d,k` (3 comma-separated values), got {}",
                    v.len()
                );
                (v[0], v[1], v[2])
            })
            .collect(),
        Err(_) => vec![
            (10_000, 16, 5),
            (50_000, 16, 5),
            (100_000, 32, 10),
            (200_000, 32, 10),
        ],
    };
    for &(n, d, k) in &ladder {
        if n > cap {
            println!("{n:>8} {d:>5} {k:>4} {:>12}", "skipped");
            continue;
        }
        let fit_s = run_config(n, d, k);
        println!("{n:>8} {d:>5} {k:>4} {fit_s:>12.6}");
    }
}

/// Attribute `fit`'s wall-clock to its two host-visible stages: the
/// device→host label round-trip + host validate/remap loop (`y.to_host` +
/// classes_ bookkeeping) vs the device-to-device training-matrix duplication
/// (`device_copy`, `mlrs_backend::prims::knn::device_copy`) — so the
/// optimization target is measured, not guessed.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_classifier_fit_stage_breakdown() {
    use mlrs_backend::prims::knn::device_copy;

    let _ = env_logger::builder().is_test(true).try_init();
    let cap = max_n();
    println!(
        "{:>8} {:>5} {:>12} {:>12}",
        "n_train", "d", "label_s", "device_copy_s"
    );
    for &(n, d) in &[
        (10_000usize, 16usize),
        (50_000, 16),
        (100_000, 32),
        (200_000, 32),
    ] {
        if n > cap {
            continue;
        }
        let (x, y) = make_classification(n, d, 42);
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

        let mut label_s = f64::INFINITY;
        let mut copy_s = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let y_host = y_dev.to_host(&mut pool);
            let mut classes: Vec<i32> = y_host.iter().map(|&v| v.round() as i32).collect();
            classes.sort_unstable();
            classes.dedup();
            std::hint::black_box(&classes);
            let e0 = t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            let copied = device_copy::<f32>(&mut pool, &x_dev);
            // Force completion with a REAL boundary readback of one element:
            // `DeviceArray::to_host` sizes its transfer off the `Handle`'s own
            // `size_in_used()` (`self.size - offset_start - offset_end`), NOT
            // off the `len` a caller wraps it with — `from_raw(handle, 1)`
            // alone still reads back the WHOLE buffer. `offset_end` trims the
            // handle's used size down to one `f32` (4 bytes) FIRST, so this
            // actually measures kernel-completion latency, not a second full
            // memcpy of the copied buffer (code-review KNN-FIT-01 fix).
            let elem_bytes = std::mem::size_of::<f32>() as u64;
            let total_bytes = copied.byte_size() as u64;
            let trimmed_handle = copied.handle().clone().offset_end(total_bytes - elem_bytes);
            let probe: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(trimmed_handle, 1);
            let _ = probe.to_host(&mut pool);
            let e1 = t1.elapsed().as_secs_f64();
            copied.release_into(&mut pool);

            if e0 < label_s {
                label_s = e0;
            }
            if e1 < copy_s {
                copy_s = e1;
            }
        }
        println!("{n:>8} {d:>5} {label_s:>12.6} {copy_s:>12.6}");
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
    }
}
