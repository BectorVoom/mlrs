//! `BayesianRidge` **fit** and **predict** wall-clock probe (BAYES-GPU) — the
//! device arms against mlrs's own host arms, on ONE machine.
//!
//! ## What is being measured, and why the A/B ladders are the point
//! "The GPU beat the CPU" is only meaningful against mlrs's own best host code,
//! not against whatever the default dispatch happened to pick — otherwise the
//! two columns come from different shapes and cannot be divided. So every
//! ladder here runs BOTH arms at EVERY rung, forced through the `abflag` knobs,
//! and prints the ratio:
//!
//! | ladder | host arm | device arm | knob |
//! |---|---|---|---|
//! | `fit` | `fit_from_host_slice`, no upload at all | upload + `f64` device Gram | `MLRS_BAYES_FIT_HOST` |
//! | `predict_std` | parallel `f64` host sweep | upload + fused kernel | `MLRS_BAYES_STD_HOST` |
//!
//! The `fit` timer starts BEFORE the upload, because a Python caller passes a
//! numpy array and the transfer is part of `fit` for them. On a T4 that transfer
//! was 85–96% of a `Ridge` device fit ([[mlrs-ridge-positive-cuda]],
//! [[mlrs-ridge-default-cuda]]), so an upload-free ladder measures something no
//! user experiences — and here it would also flatter the arm whose entire
//! premise is that the `O(n·d²)` reduction outgrows the `O(n·d)` transfer.
//!
//! ## Why this probe runs at `f64` and targets a compute-class GPU
//! `BayesianRidge` consumes its Gram through the residual identity, whose error
//! is amplified by `yᵀy/sse`, so the reduction runs in `f64` on every backend
//! (`prims::normal_eq` module docs). That makes the card's DOUBLE-precision rate
//! the relevant number, and it is not a detail: a P100 (GP100) runs `f64` at
//! 1/2 of `f32`, a T4 (TU104) at **1/32**. This estimator is one of the few in
//! the crate for which a Pascal compute card is the RIGHT accelerator and a
//! newer inference card is the wrong one.
//!
//! ```text
//! # the mlrs CPU baseline (the number the device arm has to beat)
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test bayesian_ridge_perf_test -- --ignored --nocapture
//!
//! # the CUDA number
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test bayesian_ridge_perf_test -- --ignored --nocapture
//! ```
//!
//! `BAYES_PROFILE=1` adds `bayesian_ridge.rs`'s per-phase attribution
//! (gram / eig / loop / sigma), which is what says whether a rung is
//! reduction-bound (the part this campaign moved) or eigen-bound (the `O(d³)`
//! host tail that it did not). `MLRS_BAYES_REPS` overrides the min-of-N repeat
//! count (default 5); the first config runs a discarded warmup so pipeline
//! compilation is excluded.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::bayesian_ridge::BayesianRidge;
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64, byte-identical to `ridge_default_perf_test.rs` and
/// `scripts/bench_ridge.py`, so every engine in a cross-language comparison fits
/// exactly the same dataset.
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

/// The same deterministic design the `Ridge` probes use, at `f64`.
///
/// The noise level matters here in a way it does not for `Ridge`: the evidence
/// iteration's stopping rule is `Σ|Δcoef| < tol`, and a near-noiseless design
/// converges in 3–5 iterations at sklearn's default `tol`, which barely
/// exercises the loop at all ([[mlrs-bayesian-ridge-cpu]]). `0.1` puts the fit
/// in a regime where the precisions actually move.
fn make_regression(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
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
        dot += 0.1 * uniform_pm1(&mut sn);
        y.push(dot);
    }
    (x, y)
}

/// One timed `fit`, through whichever ingress `host_fit_applicable` selects —
/// the PyO3 `bayes_fit_dispatch!` branch, reproduced so a `--features cpu`
/// number and a `--features cuda` number measure the same user-visible
/// operation.
///
/// The upload is INSIDE the timer on the device branch, and the terminal
/// `intercept(pool)` read-back is what makes the timer include every queued
/// kernel rather than just the enqueue.
fn fit_once(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
) -> (f64, &'static str, usize) {
    let est = BayesianRidge::<f64>::new();
    let host_arm = est.host_fit_applicable((n, d));

    let t0 = Instant::now();
    let (fitted, arm): (BayesianRidge<f64, Fitted>, &'static str) = if host_arm {
        (
            est.fit_from_host_slice(pool, x, y, (n, d), None)
                .expect("host fit"),
            "host",
        )
    } else {
        let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, x);
        let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, y);
        let f = est.fit(pool, &xd, Some(&yd), (n, d)).expect("device fit");
        xd.release_into(pool);
        yd.release_into(pool);
        (f, "device")
    };
    let intercept = fitted.intercept(pool);
    let elapsed = t0.elapsed().as_secs_f64();

    assert!(
        intercept.is_finite(),
        "degenerate fit at n={n} d={d} — the perf run is broken"
    );
    (elapsed, arm, fitted.n_iter())
}

/// One timed `predict(X, return_std=True)` standard deviation over host rows.
/// The read-back is inside the prim, so the returned time already covers the
/// whole launch.
fn std_once(
    pool: &mut BufferPool<ActiveRuntime>,
    fitted: &BayesianRidge<f64, Fitted>,
    xt: &[f64],
    n: usize,
    d: usize,
) -> f64 {
    let t0 = Instant::now();
    let out = fitted
        .predict_std_from_host(pool, xt, (n, d))
        .expect("predict_std");
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(out[0].is_finite() && out[0] > 0.0, "degenerate predict_std");
    elapsed
}

/// One timed `predict` mean over host rows — already a fused device kernel
/// before this campaign, printed so the `return_std` column has its natural
/// baseline next to it.
fn mean_once(
    pool: &mut BufferPool<ActiveRuntime>,
    fitted: &BayesianRidge<f64, Fitted>,
    xt: &[f64],
    n: usize,
    d: usize,
) -> f64 {
    let t0 = Instant::now();
    let out = fitted
        .predict_from_host(pool, xt, (n, d))
        .expect("predict_from_host");
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(out.operand_finite && out.values[0].is_finite());
    elapsed
}

/// The ladder. `d` is walked to 256 because the device arm's whole premise is
/// that `n·d²/2` multiply-adds outgrow an `n·d` transfer — the margin is
/// governed by `d`, not by `n`, since both terms are linear in `n`.
const CONFIGS: &[(usize, usize)] = &[
    (1_000, 8),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (100_000, 128),
    (100_000, 256),
    (200_000, 256),
];

/// Rows in the held-out matrix the predict ladders time.
const N_TEST: usize = 100_000;

fn reps() -> usize {
    std::env::var("MLRS_BAYES_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// The headline `fit` ladder: whatever `host_fit_applicable` picks, which is
/// what a Python caller gets.
fn fit_headline(pool: &mut BufferPool<ActiveRuntime>) {
    println!(
        "\n== fit: dispatch-chosen arm ==  backend={} reps={}",
        mlrs_backend::capability::active_backend_name(),
        reps()
    );
    println!(
        "{:>9} {:>5} | {:>11} | {:>6} {:>6}",
        "n", "d", "fit (ms)", "arm", "iters"
    );
    for (i, &(n, d)) in CONFIGS.iter().enumerate() {
        let (x, y) = make_regression(n, d, 42);
        if i == 0 {
            fit_once(pool, &x, &y, n, d); // warmup (JIT + first touch)
        }
        let mut best = f64::INFINITY;
        let (mut arm, mut iters) = ("", 0);
        for _ in 0..reps() {
            let (t, a, it) = fit_once(pool, &x, &y, n, d);
            best = best.min(t);
            arm = a;
            iters = it;
        }
        println!("{n:>9} {d:>5} | {:>11.3} | {arm:>6} {iters:>6}", best * 1e3);
    }
}

/// The A/B that answers "does the gpu beat the cpu" for `fit`: BOTH arms, at
/// every rung, on one machine.
fn fit_arm_ab(pool: &mut BufferPool<ActiveRuntime>) {
    let backend = mlrs_backend::capability::active_backend_name();
    if backend == "cpu" {
        println!("\n== fit: forced-arm A/B ==  backend=cpu — SKIPPED");
        println!("  The device arm is refused outright on cpu (`device_gram_applicable`):");
        println!("  a GPU-shaped reduction on a runtime that spawns one OS thread per unit");
        println!("  and JITs at -O0 is the pathology the host arm exists to avoid. The cpu");
        println!("  column to compare against is the `host` arm printed above.");
        return;
    }
    println!(
        "\n== fit: forced-arm A/B ==  backend={backend} reps={}",
        reps()
    );
    println!(
        "{:>9} {:>5} | {:>11} {:>11} | {:>9}",
        "n", "d", "host (ms)", "device (ms)", "speedup"
    );
    for (i, &(n, d)) in CONFIGS.iter().enumerate() {
        let (x, y) = make_regression(n, d, 42);
        if i == 0 {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "0");
            fit_once(pool, &x, &y, n, d);
        }

        let mut host = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "1");
            for _ in 0..reps() {
                let (t, a, _) = fit_once(pool, &x, &y, n, d);
                assert_eq!(a, "host", "the =1 force must reach the dispatcher");
                host = host.min(t);
            }
        }
        let mut dev = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "0");
            for _ in 0..reps() {
                let (t, a, _) = fit_once(pool, &x, &y, n, d);
                assert_eq!(a, "device", "the =0 force must reach the dispatcher");
                dev = dev.min(t);
            }
        }
        println!(
            "{n:>9} {d:>5} | {:>11.3} {:>11.3} | {:>8.2}x",
            host * 1e3,
            dev * 1e3,
            host / dev
        );
    }
}

/// `predict` (mean) and `predict(return_std=True)`, host arm vs device kernel.
///
/// The mean is printed alongside deliberately: it is `O(n·d)` against the
/// standard deviation's `O(n·d²)`, so the two columns show directly why
/// `return_std` is the predict path with room for a device arm and the mean is
/// not.
fn predict_arm_ab(pool: &mut BufferPool<ActiveRuntime>) {
    let backend = mlrs_backend::capability::active_backend_name();
    println!(
        "\n== predict: forced-arm A/B ==  backend={backend} reps={} n_test={N_TEST}",
        reps()
    );
    println!(
        "{:>5} | {:>10} | {:>11} {:>11} {:>9}",
        "d", "mean (ms)", "std host", "std device", "speedup"
    );

    for (i, &d) in [16usize, 64, 128, 256].iter().enumerate() {
        let (x, y) = make_regression(20_000, d, 42);
        let (xt, _) = make_regression(N_TEST, d, 7);
        let fitted = {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "1");
            BayesianRidge::<f64>::new()
                .fit_from_host_slice(pool, &x, &y, (20_000, d), None)
                .expect("fit for predict ladder")
        };
        if i == 0 {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", "0");
            std_once(pool, &fitted, &xt, N_TEST, d);
            mean_once(pool, &fitted, &xt, N_TEST, d);
        }

        let mut mean = f64::INFINITY;
        for _ in 0..reps() {
            mean = mean.min(mean_once(pool, &fitted, &xt, N_TEST, d));
        }
        let mut sh = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", "1");
            for _ in 0..reps() {
                sh = sh.min(std_once(pool, &fitted, &xt, N_TEST, d));
            }
        }
        // On the two backends `prims::linear_predict`'s `use_host_std` routes to
        // the host, forcing the device kernel measures nothing anyone runs — and
        // on both it costs far more than the information is worth.
        //
        // On **cpu** it is the cubecl-cpu pathology (one OS thread per unit,
        // `-O0` JIT): measured locally at 4.7 s per rep at `d = 256` against
        // 1.3 s for the host sweep, so the forced leg alone is minutes of a
        // probe whose useful output is one number.
        //
        // On **wgpu** it does not measure a slower arm, it KILLS THE ADAPTER: `d = 256` over 100 000 rows is
        // ~9 s of `f64` GPU time there, past the compositor's timeout, and the
        // run dies with `context is lost` + `BufferAsyncError` mid-ladder —
        // taking the cpu leg of the whole probe with it. The host row above is
        // the arm that backend actually uses (`prims::linear_predict`'s
        // `use_host_std` routes wgpu there BECAUSE it is 12–25× faster), so
        // there is no ratio to print.
        if backend == "cpu" || backend == "wgpu" {
            println!(
                "{d:>5} | {:>10.3} | {:>11.3} {:>11} {:>9}",
                mean * 1e3,
                sh * 1e3,
                "skipped",
                "host-routed"
            );
            continue;
        }
        let mut sd = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", "0");
            for _ in 0..reps() {
                sd = sd.min(std_once(pool, &fitted, &xt, N_TEST, d));
            }
        }
        println!(
            "{d:>5} | {:>10.3} | {:>11.3} {:>11.3} {:>8.2}x",
            mean * 1e3,
            sh * 1e3,
            sd * 1e3,
            sh / sd
        );
    }
}

/// The ladder that separates the GPU's ARITHMETIC from the bus: both operands
/// are uploaded ONCE, outside the timer, and only the compute is measured.
///
/// ## Why this is a real case and not a flattering one
/// The other ladders time `fit` the way a Python caller experiences it, with the
/// upload inside — which is right for them, and on a transfer-starved host it
/// makes the device arm lose no matter how fast the card is. But a Rust caller
/// running a pipeline (fit, then predict, then score, over a design that never
/// leaves the device) pays that transfer once for MANY operations, or not at
/// all if the design was produced on the device. That caller reaches
/// [`Fit::fit`] and [`BayesianRidge::predict_std`] with `DeviceArray`s already
/// in hand, and `host_fit_applicable` is not consulted on that path at all.
///
/// So this ladder answers the question the other two cannot: given the data is
/// already there, is the device arm worth having? Read together, the two
/// numbers also decompose the device column — the difference between this
/// ladder and the forced-device column above IS the transfer, which is how the
/// bus bandwidth gets measured without a separate probe.
fn resident_ladder(pool: &mut BufferPool<ActiveRuntime>) {
    let backend = mlrs_backend::capability::active_backend_name();
    if backend == "cpu" {
        println!("\n== device-resident operands ==  backend=cpu — SKIPPED (no device)");
        return;
    }
    println!(
        "\n== device-resident operands (upload EXCLUDED from the timer) ==  \
         backend={backend} reps={}",
        reps()
    );
    println!(
        "{:>9} {:>5} | {:>11} {:>11} | {:>9}",
        "n", "d", "host (ms)", "device (ms)", "speedup"
    );
    for &(n, d) in CONFIGS {
        let (x, y) = make_regression(n, d, 42);
        let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, &x);
        let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, &y);

        let mut host = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "1");
            for _ in 0..reps() {
                let (t, _, _) = fit_once(pool, &x, &y, n, d);
                host = host.min(t);
            }
        }

        let mut dev = f64::INFINITY;
        for _ in 0..reps() {
            let t0 = Instant::now();
            let fitted = BayesianRidge::<f64>::new()
                .fit(pool, &xd, Some(&yd), (n, d))
                .expect("device-resident fit");
            // The fitted state is device-resident (D-03); this one-element read
            // is what makes the timer include every queued kernel rather than
            // just the enqueue.
            let i = fitted.intercept(pool);
            let t = t0.elapsed().as_secs_f64();
            assert!(i.is_finite(), "degenerate resident fit at n={n} d={d}");
            dev = dev.min(t);
        }
        println!(
            "{n:>9} {d:>5} | {:>11.3} {:>11.3} | {:>8.2}x",
            host * 1e3,
            dev * 1e3,
            host / dev
        );
        xd.release_into(pool);
        yd.release_into(pool);
    }

    // predict(return_std=True) over a design that is already resident — the
    // `O(n·d²)` path, where the arithmetic-to-transfer ratio is highest.
    println!(
        "\n{:>5} | {:>11} {:>11} | {:>9}   (predict_std, resident)",
        "d", "host (ms)", "device (ms)", "speedup"
    );
    for &d in &[16usize, 64, 128, 256] {
        let (x, y) = make_regression(20_000, d, 42);
        let (xt, _) = make_regression(N_TEST, d, 7);
        let fitted = {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_FIT_HOST", "1");
            BayesianRidge::<f64>::new()
                .fit_from_host_slice(pool, &x, &y, (20_000, d), None)
                .expect("fit for resident predict ladder")
        };
        let xtd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, &xt);

        let mut host = f64::INFINITY;
        {
            let _g = mlrs_backend::abflag::force("MLRS_BAYES_STD_HOST", "1");
            for _ in 0..reps() {
                host = host.min(std_once(pool, &fitted, &xt, N_TEST, d));
            }
        }
        let mut dev = f64::INFINITY;
        for _ in 0..reps() {
            let t0 = Instant::now();
            let out = fitted
                .predict_std(pool, &xtd, (N_TEST, d))
                .expect("resident predict_std");
            let v = out.to_host(pool);
            let t = t0.elapsed().as_secs_f64();
            assert!(v[0].is_finite() && v[0] > 0.0, "degenerate resident std");
            out.release_into(pool);
            dev = dev.min(t);
        }
        println!(
            "{d:>5} | {:>11.3} {:>11.3} | {:>8.2}x",
            host * 1e3,
            dev * 1e3,
            host / dev
        );
        xtd.release_into(pool);
    }
}

/// All four ladders, in ONE `#[test]` — they share the pool, and splitting them
/// would pay the runtime's first-touch cost per ladder.
#[test]
#[ignore = "perf probe: cargo test --release -- --ignored --nocapture"]
fn bayesian_ridge_perf_ladders() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    // BOTH f64 flags, because on cuda they disagree and only the second one
    // governs whether the device arm engages (see
    // `capability::f64_device_kernels_available`). A ladder printed without
    // them cannot be read: an all-`host` arm column could mean "the device lost"
    // or "the device arm was never legal here".
    println!(
        "mlrs BayesianRidge perf — backend={} dtype=f64 f64_advertised={} f64_runnable={}",
        mlrs_backend::capability::active_backend_name(),
        mlrs_backend::capability::feature_enabled(mlrs_backend::capability::FloatKind::F64),
        mlrs_backend::capability::f64_device_kernels_available(),
    );
    fit_headline(&mut pool);
    fit_arm_ab(&mut pool);
    predict_arm_ab(&mut pool);
    resident_ladder(&mut pool);
}
