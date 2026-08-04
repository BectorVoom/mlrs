//! Ridge `predict` — DEVICE kernel vs mlrs's OWN cpu host-native matvec,
//! same process, same hardware (RIDGE-PREDICT-CUDA-VS-CPU).
//!
//! Every prior predict-perf campaign
//! (`mlrs-linear-predict-optimization`/`mlrs-linear-predict-coalesced` project
//! memory) measured the device kernel against cuML/sklearn on a T4 and found it
//! 7-50x faster — but never against mlrs's OWN zero-copy cpu predict path
//! (`prims::linear_predict::linear_predict_host`, added later, LINEAR-PRED-CPU),
//! which is a `-O3`-vectorized host matvec with NO upload at all. On the FIT
//! side, the equivalent comparison (`mlrs-ridge-default-cuda` memory) found the
//! device arm does NOT always beat mlrs's own host arm — the crossover depends
//! on `d`, because the GPU's advantage is `n·d²` arithmetic over an `n·d`
//! transfer. `predict` is `O(n·d)` compute over the SAME `O(n·d)` transfer, a
//! strictly WORSE compute-to-transfer ratio than fit's `O(n·d²)`, so this
//! comparison is genuinely unresolved going in, not a formality.
//!
//! Both arms run in ONE process on the SAME machine: `linear_predict_host` /
//! `linear_predict_multi_host` are backend-agnostic free functions (the exact
//! code EVERY build's `Ridge::predict_from_host` calls now, cpu or not — see
//! the RESULT below), so calling them directly from a `--features cuda` (or
//! any other) test binary reproduces the cpu arm's arithmetic without a second
//! build or a second machine — the same technique the fit-side campaigns used
//! via forced A/B dispatch.
//!
//! ## RESULT (Kaggle P100, single-target, `n` 10k-1M, `d` 16-64): device LOSES
//! 10-23x. `n_targets=4`: loses 2-3x. `n_targets=16`: wins marginally (~1.1-
//! 1.5x, two data points, close to a wash at `d=64` — too thin to act on).
//! `Ridge::predict_from_host`/`predict_multi_from_host` now take the HOST arm
//! UNCONDITIONALLY on every backend (`ridge.rs`'s RIDGE-PREDICT-CUDA-VS-CPU
//! section), so the "device" column below measures the RAW device kernel via
//! `Predict::predict` (upload + fused GATHER launch + read back) directly —
//! the production entry point no longer takes that arm, so it cannot be
//! measured through it anymore. This keeps the probe meaningful for
//! re-verification if a future device-side lever changes the picture.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test ridge_predict_device_vs_host_perf_test -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` by default; a wall-clock probe, not part of the ordinary suite.
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use bytemuck::Pod;

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::{linear_predict_host, linear_predict_multi_host};
use mlrs_backend::runtime::{self, ActiveRuntime};

const REPS: usize = 7;

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

/// `X` uniform `[-1,1)^d` (seed 42), `y[:, t] = X @ coef_t + 0.5 + 0.01*noise_t`
/// (`t` independent coefficient streams), byte-identical convention to
/// `ridge_perf_test.rs` / `ridge_multi_target_test.rs`.
fn make_multi_regression(n: usize, d: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sx = 42u64;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();
    let mut y = vec![0.0f64; n * t];
    for target in 0..t {
        let mut sc = 43 + (target as u64) * 2;
        let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();
        for r in 0..n {
            let mut dot = 0.5;
            for c in 0..d {
                dot += x[r * d + c] * coef[c];
            }
            y[r * t + target] = dot;
        }
    }
    (
        x.iter().map(|&v| v as f32).collect(),
        y.iter().map(|&v| v as f32).collect(),
    )
}

fn min_secs(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("f32/f64 only"),
    }
}

/// One `(n, d, n_targets)` config: fit on-device, then time the RAW DEVICE
/// kernel (`Predict::predict` — upload + fused GATHER launch + read back, the
/// arm the production `predict_from_host` entry point no longer takes) against
/// the SAME arithmetic run directly on the host (`linear_predict_host`/
/// `linear_predict_multi_host` — what every build, cpu or not, now actually
/// executes via `predict_from_host`/`predict_multi_from_host`).
/// `coef`/`bias` are read to host ONCE, outside the timed region, mirroring the
/// `HostMirror` memoization every production predict call already benefits
/// from.
fn run_config(n: usize, d: usize, n_targets: usize) -> (f64, f64) {
    let (x, y) = make_multi_regression(n, d, n_targets);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let fitted = if n_targets == 1 {
        Ridge::<f32>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .build::<f32>()
            .expect("build")
            .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
            .expect("fit")
    } else {
        Ridge::<f32>::builder()
            .alpha(1.0)
            .fit_intercept(true)
            .build::<f32>()
            .expect("build")
            .fit_multi_target_with_sample_weight(
                &mut pool, &x_dev, &y_dev, (n, d), n_targets, None,
            )
            .expect("multi-target fit")
    };
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);

    // A fresh test matrix (not the training design), read as a host slice —
    // exactly the shape both `predict_from_host` and `linear_predict_host`
    // consume, so neither arm gets a home-field advantage on ingress.
    let mut sxt = 999u64;
    let x_test: Vec<f32> = (0..n * d).map(|_| f64_to::<f32>(uniform_pm1(&mut sxt))).collect();

    if n_targets == 1 {
        let coef = fitted.coef(&pool);
        let bias = fitted.intercept(&pool);

        // The RAW device kernel: upload x_test + run `Predict::predict`'s
        // fused GATHER launch + read back — the exact cost the OLD
        // "always device on non-cpu backends" dispatch paid on every call.
        // `Ridge::predict_from_host` no longer takes this arm (RIDGE-
        // PREDICT-CUDA-VS-CPU), so it is measured directly here rather than
        // through the (now host-only) production entry point, keeping this
        // probe meaningful for future re-verification.
        let device_s = min_secs(|| {
            let x_dev_test: DeviceArray<ActiveRuntime, f32> =
                DeviceArray::from_host(&mut pool, &x_test);
            let pred = fitted
                .predict(&mut pool, &x_dev_test, (n, d))
                .expect("device predict");
            let host = pred.to_host(&pool);
            x_dev_test.release_into(&mut pool);
            pred.release_into(&mut pool);
            assert_eq!(host.len(), n, "degenerate predict — perf run is broken");
        });
        let host_s = min_secs(|| {
            let pred =
                linear_predict_host::<f32>(&x_test, &coef, bias, (n, d)).expect("host predict");
            assert_eq!(pred.values.len(), n, "degenerate predict — perf run is broken");
        });
        (device_s, host_s)
    } else {
        let coef = fitted.coef_multi(&pool);
        let bias = fitted.intercept_multi(&pool);

        // Same rationale as the single-target arm above: the raw device
        // kernel (upload + `Predict::predict` + readback), not the (now
        // host-only) `predict_multi_from_host` production entry point.
        let device_s = min_secs(|| {
            let x_dev_test: DeviceArray<ActiveRuntime, f32> =
                DeviceArray::from_host(&mut pool, &x_test);
            let pred = fitted
                .predict(&mut pool, &x_dev_test, (n, d))
                .expect("device predict");
            let host = pred.to_host(&pool);
            x_dev_test.release_into(&mut pool);
            pred.release_into(&mut pool);
            assert_eq!(
                host.len(),
                n * n_targets,
                "degenerate multi-target predict — perf run is broken"
            );
        });
        let host_s = min_secs(|| {
            let pred = linear_predict_multi_host::<f32>(&x_test, &coef, &bias, (n, d), n_targets)
                .expect("host multi-target predict");
            assert_eq!(
                pred.values.len(),
                n * n_targets,
                "degenerate multi-target predict — perf run is broken"
            );
        });
        (device_s, host_s)
    }
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn ridge_predict_device_vs_host_ladder() {
    // (n_samples, n_features, n_targets). n_targets=1 rows are directly
    // comparable to `ridge_perf_test.rs`'s ladder; n_targets>1 rows exercise the
    // NEW multi-target predict kernel this campaign adds.
    let configs: &[(usize, usize, usize)] = &[
        (10_000, 16, 1),
        (10_000, 64, 1),
        (100_000, 16, 1),
        (100_000, 64, 1),
        (500_000, 16, 1),
        (1_000_000, 16, 1),
        (100_000, 16, 4),
        (100_000, 64, 4),
        (100_000, 16, 16),
        (100_000, 64, 16),
    ];
    println!(
        "{:>9} {:>4} {:>3} | {:>12} {:>12} {:>8}",
        "n", "d", "t", "device (s)", "host (s)", "device/host"
    );
    // Warmup: first config runs once, discarded, so pipeline/JIT compilation is
    // excluded from the steady-state numbers.
    let &(n0, d0, t0) = &configs[0];
    run_config(n0, d0, t0);
    for &(n, d, t) in configs {
        let (device_s, host_s) = run_config(n, d, t);
        let ratio = device_s / host_s;
        println!("{n:>9} {d:>4} {t:>3} | {device_s:>12.6} {host_s:>12.6} {ratio:>8.3}");
    }
}
