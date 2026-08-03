//! `RidgeClassifier` ON-DEVICE vs HOST wall-clock probe (RIDGECLF-CUDA).
//!
//! A plain `std::time::Instant` probe (the `ridge_default_perf_test.rs`
//! precedent — NOT a Criterion micro-benchmark). `#[ignore]` by default; run
//! TARGETED in release mode on the backend you are gating:
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test ridge_classifier_cuda_perf_test -- --ignored --nocapture
//! ```
//!
//! ## What it compares, and why that is the right comparison
//! Both arms start from the SAME host-resident operands and are timed
//! end-to-end with the upload INSIDE the timer, because that is what a Python
//! caller actually pays: the design arrives over Arrow, on the host, every
//! time. The `host` column is literally the code the **cpu** backend runs
//! (`fit_from_host_slice` / `predict_labels_from_host`), so a ratio above 1.0
//! on a cuda build is the device arm beating this crate's own cpu path on the
//! same machine — which is the claim, and it is the only version of the claim
//! that survives being read carefully. (A cuda box's CPU is not necessarily the
//! CPU a cpu-backend user has; see the `mlrs-ridge-default-cuda` note on
//! Colab's 2-vCPU Xeon flattering the GPU by ~4×.)
//!
//! ## Both ladders live in ONE `#[test]`
//! `libtest` runs a binary's tests on PARALLEL threads, so two wall-clock
//! probes in separate `#[test]`s interleave into numbers that are neither's
//! (measured on a T4: a headline 216.9 ms against the same fit's 184.1 ms in a
//! sibling ladder). One test function, run sequentially.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge_classifier::RidgeClassifier;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 — byte-identical to `scripts/bench_ridge.py` and
/// every other perf probe here, so the Python and Rust runs fit the same bytes.
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

/// `n × d` design with a class-dependent shift on one feature, and the matching
/// length-`n` label vector over `k_classes` labels. Separable enough that the
/// fit is non-degenerate; the shape, not the separability, is what is timed.
fn make_classification(n: usize, d: usize, k_classes: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut x = vec![0.0f32; n * d];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let ci = r % k_classes;
        for c in 0..d {
            let shift = if c == ci % d { 1.5 } else { 0.0 };
            x[r * d + c] = (uniform_pm1(&mut s) + shift) as f32;
        }
        y[r] = ci as f32;
    }
    (x, y)
}

fn reps() -> usize {
    std::env::var("MLRS_RIDGECLF_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// One FIT through the fused DEVICE arm, upload included.
fn fit_device(pool: &mut BufferPool<ActiveRuntime>, x: &[f32], y: &[f32], n: usize, d: usize) -> f64 {
    let est = RidgeClassifier::<f32>::new();
    let t0 = Instant::now();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(pool, x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(pool, y);
    let fitted = est
        .fit_with_sample_weight(pool, &x_dev, Some(&y_dev), (n, d), None)
        .expect("device fit");
    // The fitted state is device-resident (D-03); this read is what makes the
    // timer include every queued kernel rather than just the enqueue.
    let ic = fitted.intercept(pool);
    let dt = t0.elapsed().as_secs_f64();
    assert!(
        ic.iter().all(|v| v.is_finite()),
        "degenerate fit at n={n} d={d} — the perf run is broken"
    );
    x_dev.release_into(pool);
    y_dev.release_into(pool);
    dt
}

/// One FIT through the shared-Gram HOST arm — the code the cpu backend runs.
///
/// `host_fit_applicable` is FALSE on a device backend above `gram_host`'s
/// fixed dispatch-cost floor, which is every interesting rung of this ladder —
/// so the arm has to be FORCED for the comparison to exist at all. Without
/// this the host column reads `inf` at seven of eight rungs and the ratio is
/// vacuous. The force goes through `abflag`'s thread-local override, never
/// `std::env::set_var` (an environ data race, and it would leak across the
/// other tests in the binary).
fn fit_host(pool: &mut BufferPool<ActiveRuntime>, x: &[f32], y: &[f32], n: usize, d: usize) -> f64 {
    let _forced = mlrs_backend::abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
    let est = RidgeClassifier::<f32>::new();
    if !est.host_fit_applicable((n, d)) {
        return f64::NAN;
    }
    let t0 = Instant::now();
    let fitted = est
        .fit_from_host_slice(pool, x, y, (n, d), None)
        .expect("host fit");
    let ic = fitted.intercept(pool);
    let dt = t0.elapsed().as_secs_f64();
    assert!(ic.iter().all(|v| v.is_finite()));
    dt
}

/// One PREDICT through the fused DEVICE classify kernel, upload included (the
/// query starts on the host — see the module docs).
fn predict_device(
    pool: &mut BufferPool<ActiveRuntime>,
    est: &RidgeClassifier<f32, mlrs_algos::typestate::Fitted>,
    xq: &[f32],
    m: usize,
    d: usize,
) -> f64 {
    let t0 = Instant::now();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(pool, xq);
    let labels = est
        .predict_labels_device(pool, &xq_dev, (m, d))
        .expect("device predict");
    let out = labels.to_host(pool);
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(out.len(), m);
    xq_dev.release_into(pool);
    labels.release_into(pool);
    dt
}

/// One PREDICT through the no-upload HOST matvec — the cpu backend's arm.
fn predict_host(
    pool: &BufferPool<ActiveRuntime>,
    est: &RidgeClassifier<f32, mlrs_algos::typestate::Fitted>,
    xq: &[f32],
    m: usize,
    d: usize,
) -> f64 {
    let t0 = Instant::now();
    let pred = est
        .predict_labels_from_host(pool, xq, (m, d))
        .expect("host predict");
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(pred.labels.len(), m);
    dt
}

/// Count the rows where the two `predict` arms chose different labels, and
/// PANIC on any such row that is not a genuine near-tie.
///
/// The device kernel accumulates each class score in `F`; the host one widens
/// every operand to `f64`. Two classes whose scores sit within `f32` rounding
/// of each other can therefore order differently between the arms without
/// either being wrong — but a disagreement between two CLEARLY separated
/// scores would be a real defect (a wrong coefficient layout, a bad
/// `classes_` index, an argmax that scans the wrong range). This separates the
/// two by checking, against the host's own `f64` scores, that the top two
/// classes of a disagreeing row are within `f32` epsilon of each other.
///
/// `scores` is the host `decision_function` output — length `n_query` when
/// `n_targets == 1`, else row-major `n_query × n_targets`.
#[allow(clippy::too_many_arguments)]
fn disagreements(
    device: &[i32],
    host: &[i32],
    scores: &[f64],
    n_targets: usize,
    n: usize,
    d: usize,
    k: usize,
) -> usize {
    // f32 has ~7 significant digits; a dot over `d` terms accumulates a few
    // ulps per term, so the bound is scaled by the score magnitude and by the
    // reduction length rather than being an absolute constant.
    let rel = 1e-6 * (d as f64).sqrt();
    let mut count = 0usize;
    for (r, (&dl, &hl)) in device.iter().zip(host.iter()).enumerate() {
        if dl == hl {
            continue;
        }
        count += 1;
        let row = &scores[r * n_targets..(r + 1) * n_targets];
        let gap = if n_targets == 1 {
            // Binary: the "tie" is the single score against the 0 threshold.
            row[0].abs()
        } else {
            let mut sorted: Vec<f64> = row.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).expect("scores are finite"));
            sorted[0] - sorted[1]
        };
        // Relative to the row's own magnitude, floored at 1 so a row of
        // near-zero scores still gets an absolute f32-epsilon allowance.
        let scale = row.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        assert!(
            gap <= rel * scale,
            "n={n} d={d} k={k} row {r}: the arms chose different labels \
             ({dl} device, {hl} host) but the top-two score gap is {gap:e}, \
             far above f32 rounding ({:e}) — this is a DEFECT, not a tie",
            rel * scale
        );
    }
    count
}

/// `(n_samples, n_features, n_classes)`. The `d = 128` / `d = 256` rungs are
/// where a device FIT has a chance at all (`n·d²/2` of arithmetic over an `n·d`
/// transfer, so the advantage grows with `d`); the `k = 26` rungs are where a
/// device PREDICT does (`k`× the compute over the same transfer, and `k`× less
/// egress).
const CONFIGS: &[(usize, usize, usize)] = &[
    (10_000, 16, 2),
    (10_000, 64, 3),
    (100_000, 16, 3),
    (100_000, 16, 26),
    (100_000, 64, 3),
    (100_000, 64, 5),
    (100_000, 64, 10),
    (100_000, 64, 26),
    (100_000, 128, 10),
    (100_000, 128, 26),
    (100_000, 256, 10),
    (100_000, 256, 26),
];

/// Query rows for the predict ladder. Deliberately LARGE: a sub-millisecond
/// predict on a busy machine measures fixed overhead, not the kernel — the
/// LINEAR-07 cpu campaign concluded a regression from an `n_query = 1000`
/// ladder that reversed at a realistic batch size.
const N_QUERY: usize = 100_000;

#[test]
#[ignore = "wall-clock probe; run explicitly in release mode"]
fn ridge_classifier_device_vs_host_ladder() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let r = reps();

    println!(
        "\nRidgeClassifier FIT — min-of-{r}, f32, upload INSIDE the timer, both arms forced"
    );
    println!("{:>9} {:>5} {:>4} {:>12} {:>12} {:>9}", "n", "d", "k", "host (ms)", "device (ms)", "speedup");
    for &(n, d, k) in CONFIGS {
        let (x, y) = make_classification(n, d, k, 42);
        let mut best_dev = f64::INFINITY;
        let mut best_host = f64::INFINITY;
        for _ in 0..r {
            best_dev = best_dev.min(fit_device(&mut pool, &x, &y, n, d));
            let h = fit_host(&mut pool, &x, &y, n, d);
            if h.is_finite() {
                best_host = best_host.min(h);
            }
        }
        let ratio = best_host / best_dev;
        println!(
            "{n:>9} {d:>5} {k:>4} {:>12.3} {:>12.3} {:>8.2}x",
            best_host * 1e3,
            best_dev * 1e3,
            ratio
        );
    }

    println!(
        "\nRidgeClassifier PREDICT — min-of-{r}, f32, n_query={N_QUERY}, upload INSIDE the timer"
    );
    println!("{:>9} {:>5} {:>4} {:>12} {:>12} {:>9}", "n_fit", "d", "k", "host (ms)", "device (ms)", "speedup");
    for &(n, d, k) in CONFIGS {
        let (x, y) = make_classification(n, d, k, 42);
        let (xq, _) = make_classification(N_QUERY, d, k, 4242);
        let est = RidgeClassifier::<f32>::new();
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
        let fitted = est
            .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (n, d), None)
            .expect("fit for the predict ladder");
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);

        // Agreement first: a faster arm that disagrees is not a faster arm.
        //
        // The gate is NOT exact label equality, because that is not a contract
        // either arm can honour. The device kernel accumulates each class score
        // in `F` (`f32` here) and the host one widens to `f64`, so two classes
        // whose scores sit within `f32` rounding of each other can ORDER
        // differently between the arms — measured on a T4 at
        // `d = 16, k = 26`: exactly 1 row of 100 000. The `argmax` logic itself
        // is identical (strict `>`, first-occurrence tie-break, asserted by
        // `ridge_classifier_device_test.rs`), so what has to hold is that every
        // disagreement is a genuine near-tie — which is checked here against
        // the host's own `f64` scores, not assumed.
        {
            let hp = fitted
                .predict_labels_from_host(&pool, &xq, (N_QUERY, d))
                .expect("host predict");
            let hs = fitted
                .decision_function_from_host(&pool, &xq, (N_QUERY, d))
                .expect("host decision_function");
            let xq_dev: DeviceArray<ActiveRuntime, f32> =
                DeviceArray::from_host(&mut pool, &xq);
            let dp = fitted
                .predict_labels_device(&mut pool, &xq_dev, (N_QUERY, d))
                .expect("device predict");
            let dp = dp.to_host(&pool);
            let mismatches = disagreements(&dp, &hp.labels, &hs.values, hs.n_targets, n, d, k);
            // A rate this far above `f32` rounding would mean a real defect,
            // not a tie: at `k = 26` a T4 produced ONE.
            assert!(
                mismatches * 1000 <= N_QUERY,
                "n={n} d={d} k={k}: {mismatches} of {N_QUERY} rows disagree — \
                 too many to be f32/f64 near-ties"
            );
            if mismatches > 0 {
                println!(
                    "    (n={n} d={d} k={k}: {mismatches}/{N_QUERY} near-tie label \
                     disagreements, all verified within f32 rounding)"
                );
            }
            xq_dev.release_into(&mut pool);
        }

        let mut best_dev = f64::INFINITY;
        let mut best_host = f64::INFINITY;
        for _ in 0..r {
            best_dev = best_dev.min(predict_device(&mut pool, &fitted, &xq, N_QUERY, d));
            best_host = best_host.min(predict_host(&pool, &fitted, &xq, N_QUERY, d));
        }
        println!(
            "{n:>9} {d:>5} {k:>4} {:>12.3} {:>12.3} {:>8.2}x",
            best_host * 1e3,
            best_dev * 1e3,
            best_host / best_dev
        );
    }
    println!();
}
