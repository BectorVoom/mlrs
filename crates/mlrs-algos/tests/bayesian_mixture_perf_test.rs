//! `BayesianGaussianMixture` (MIX-02) wall-clock probe — mlrs against sklearn.
//!
//! The Rust half of a two-process comparison whose other half is
//! `scripts/bench_bgm.py`. Both build the SAME design from the same
//! counter-based splitmix64 stream (shared verbatim with the MIX-01 probe), fit
//! with the same hyperparameters, and print the same columns, so the two tables
//! can be divided rung by rung.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test bayesian_mixture_perf_test -- --ignored --nocapture
//! python3 scripts/bench_bgm.py
//! ```
//!
//! ## Which parameters get a ladder, and why
//! Only the ones that change what the fit COSTS — a parameter that merely
//! shifts a number is not worth a measurement:
//!
//! | ladder | axis | why it matters |
//! |---|---|---|
//! | `cov=*` | `covariance_type` | `full`/`tied` are `O(n·k·d²)`, `diag`/`spherical` `O(n·k·d)`. The biggest lever, and where the hoisted `tied` E-step shows up. |
//! | `d=*` | `n_features` | the quadratic axis; also the axis the `O(d)` digamma sum per component rides on |
//! | `k=*` | `n_components` | linear in the sweep, but it multiplies the quadratic term — and for THIS estimator it is the parameter users deliberately over-provision, since the prior prunes |
//! | `n=*` | `n_samples` | linear; also where the worker pool starts paying |
//! | `init=*` | `init_params` | `kmeans` runs a whole Lloyd fit before the loop; the sparse routes run none but need far more iterations |
//! | `prior=*` | `weight_concentration_prior_type` | the one string parameter whose cost is NOT obvious: the stick-breaking branch adds a `k`-length scan plus `k` `betaln`s per iteration against an `O(n·k·d²)` sweep, so the ladder exists to show it is free rather than to assume it |
//! | `n_init=*` | restart count | a pure multiplier — checks the restart loop leaks no per-restart setup |
//! | `vb …` | the loop in isolation | `tol = 0` + fixed `max_iter` removes the initialization and the convergence rate from the timer, leaving only the E-step and M-step |
//!
//! The `vb` ladder is the one that attributes a win. Everything above it runs
//! to a `tol`, so a difference there can come from converging in fewer
//! iterations rather than from cheaper iterations; `n_iter` is printed on both
//! sides precisely so that confound is visible rather than hidden.
//!
//! ## Methodology
//! min-of-N after a discarded warmup, with BOTH wall and process-CPU time
//! reported. Two prior campaigns had a verdict inverted by a co-tenant job on
//! the same box ([[mlrs-cpu-bench-separate-processes]],
//! [[mlrs-bench-verify-knob-is-live]]), so the CPU column is not decoration: if
//! one engine's `cpu/wall` is far above 1 and the other's is near 1, the
//! wall-clock ratio is reporting thread count rather than efficiency.
//!
//! `MLRS_BGM_REPS` overrides the repeat count; `MLRS_GMM_UNITS` forces the
//! worker-pool width (the shared `gmm_host` `abflag` knob).
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::mixture::bayesian_gaussian_mixture::BayesianGaussianMixture;
use mlrs_algos::typestate::Fitted;
use mlrs_backend::abflag;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64, byte-identical to `scripts/bench_bgm.py` (and to
/// the MIX-01 probe's, so the two estimators are timed on the same design).
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

/// `k` well-separated isotropic blobs — the same array `bench_bgm.py` builds.
fn make_blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut centers = vec![0.0f64; k * d];
    for slot in centers.iter_mut() {
        *slot = (uniform01(&mut s) * 2.0 - 1.0) * 10.0;
    }
    let mut x = vec![0.0f64; n * d];
    for i in 0..n {
        let c = i % k;
        for j in 0..d {
            let u1 = uniform01(&mut s).max(f64::MIN_POSITIVE);
            let u2 = uniform01(&mut s);
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            x[i * d + j] = centers[c * d + j] + g;
        }
    }
    x
}

/// The two probes MUST build the same design, or the tables cannot be divided.
///
/// A plain test, not an `#[ignore]`d probe: a silent drift between this
/// `make_blobs` and `scripts/bench_bgm.py`'s would turn every published ratio
/// into a comparison of two different problems, and nothing else in the suite
/// would notice. The reference values are the MIX-01 probe's, which is the
/// point — the two estimators are benchmarked on one design.
#[test]
fn shared_design_matches_the_python_probe() {
    let got = make_blobs(4, 3, 2, 42);
    let want = [
        5.376918011619,
        -7.882205097444,
        -6.206825485956,
        -4.261804099267,
        -8.978946090085,
        5.235874905861,
        4.016352812844,
        -6.537024689807,
        -5.165802603336,
        -3.509310159428,
        -9.068499958212,
        7.245708178618,
    ];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-9,
            "make_blobs diverged from scripts/bench_bgm.py at {i}: {g} vs {w}"
        );
    }
}

fn reps() -> usize {
    std::env::var("MLRS_BGM_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// min-of-N `(wall_ms, cpu_ms)` after one discarded warmup.
fn measure<T>(mut f: impl FnMut() -> T) -> (f64, f64) {
    let _ = f();
    let mut best_wall = f64::INFINITY;
    let mut best_cpu = f64::INFINITY;
    for _ in 0..reps() {
        let c0 = cpu_time_ms();
        let t0 = Instant::now();
        let out = f();
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let cpu = cpu_time_ms() - c0;
        std::hint::black_box(out);
        best_wall = best_wall.min(wall);
        best_cpu = best_cpu.min(cpu);
    }
    (best_wall, best_cpu)
}

/// Process CPU time in milliseconds, read from `/proc/self/stat` (utime+stime).
///
/// Meaningful here because there is no cubecl in the loop at all (this
/// estimator's engine is pure host code), unlike the trap
/// [[mlrs-bench-verify-knob-is-live]] documents where cubecl's spinning threads
/// charge idle time as work.
fn cpu_time_ms() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return f64::NAN;
    };
    let Some((_, tail)) = stat.rsplit_once(')') else {
        return f64::NAN;
    };
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let utime: f64 = fields.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = fields.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) / 100.0 * 1e3
}

/// One rung of a ladder.
struct Rung {
    label: String,
    n: usize,
    d: usize,
    k: usize,
    cov: &'static str,
    init: &'static str,
    prior: &'static str,
    max_iter: usize,
    n_init: usize,
    /// `0.0` forces the loop to run exactly `max_iter` iterations in BOTH
    /// engines, which is what isolates the variational loop from the
    /// initialization AND from the convergence rate.
    tol: f64,
}

#[allow(clippy::too_many_arguments)]
fn rung(
    label: impl Into<String>,
    n: usize,
    d: usize,
    k: usize,
    cov: &'static str,
    init: &'static str,
    prior: &'static str,
    max_iter: usize,
    n_init: usize,
) -> Rung {
    Rung {
        label: label.into(),
        n,
        d,
        k,
        cov,
        init,
        prior,
        max_iter,
        n_init,
        tol: 1e-3,
    }
}

/// A rung that runs a FIXED number of iterations (`tol = 0`).
#[allow(clippy::too_many_arguments)]
fn rung_fixed_iters(
    label: impl Into<String>,
    n: usize,
    d: usize,
    k: usize,
    cov: &'static str,
    init: &'static str,
    prior: &'static str,
    max_iter: usize,
) -> Rung {
    Rung {
        tol: 0.0,
        ..rung(label, n, d, k, cov, init, prior, max_iter, 1)
    }
}

fn fit(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    r: &Rung,
) -> BayesianGaussianMixture<f64, Fitted> {
    BayesianGaussianMixture::<f64>::builder()
        .n_components(r.k)
        .covariance_type(r.cov)
        .init_params(r.init)
        .weight_concentration_prior_type(r.prior)
        .max_iter(r.max_iter)
        .n_init(r.n_init)
        .tol(r.tol)
        .reg_covar(1e-6)
        .random_state(Some(0))
        .build::<f64>()
        .expect("valid hyperparameters")
        .fit_from_host_slice(pool, x, (r.n, r.d))
        .expect("fit")
}

fn run_ladder(title: &str, rungs: &[Rung]) {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    println!("\n=== {title} ===");
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>7}",
        "rung", "fit ms", "cpu ms", "pred ms", "score ms", "n_iter"
    );
    println!("{}", "-".repeat(78));
    for r in rungs {
        let x = make_blobs(r.n, r.d, r.k, 42);
        let (fit_wall, fit_cpu) = measure(|| fit(&mut pool, &x, r));
        let fitted = fit(&mut pool, &x, r);
        let (pred_wall, _) = measure(|| fitted.predict_labels_host(&x, (r.n, r.d)).unwrap());
        let (score_wall, _) = measure(|| fitted.score_samples_host(&x, (r.n, r.d)).unwrap());
        println!(
            "{:<26} {fit_wall:>10.2} {fit_cpu:>10.2} {pred_wall:>10.2} {score_wall:>10.2} {:>7}",
            r.label,
            fitted.n_iter()
        );
    }
}

/// The full comparison ladder. `--ignored --nocapture`, release only.
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn bgm_perf_ladders() {
    const DP: &str = "dirichlet_process";
    run_ladder(
        "covariance_type (the complexity-class lever)",
        &[
            rung("cov=full", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("cov=tied", 20_000, 16, 8, "tied", "kmeans", DP, 100, 1),
            rung("cov=diag", 20_000, 16, 8, "diag", "kmeans", DP, 100, 1),
            rung(
                "cov=spherical",
                20_000,
                16,
                8,
                "spherical",
                "kmeans",
                DP,
                100,
                1,
            ),
        ],
    );
    run_ladder(
        "n_features (the quadratic axis)",
        &[
            rung("d=4", 20_000, 4, 8, "full", "kmeans", DP, 100, 1),
            rung("d=16", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("d=64", 20_000, 64, 8, "full", "kmeans", DP, 100, 1),
            rung("d=128", 20_000, 128, 8, "full", "kmeans", DP, 100, 1),
        ],
    );
    run_ladder(
        "n_components (over-provisioned on purpose — the prior prunes)",
        &[
            rung("k=2", 20_000, 16, 2, "full", "kmeans", DP, 100, 1),
            rung("k=8", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("k=32", 20_000, 16, 32, "full", "kmeans", DP, 100, 1),
        ],
    );
    run_ladder(
        "n_samples",
        &[
            rung("n=2000", 2_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("n=20000", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("n=200000", 200_000, 16, 8, "full", "kmeans", DP, 100, 1),
        ],
    );
    run_ladder(
        "init_params",
        &[
            rung("init=kmeans", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung(
                "init=k-means++",
                20_000,
                16,
                8,
                "full",
                "k-means++",
                DP,
                100,
                1,
            ),
            rung("init=random", 20_000, 16, 8, "full", "random", DP, 100, 1),
            rung(
                "init=random_from_data",
                20_000,
                16,
                8,
                "full",
                "random_from_data",
                DP,
                100,
                1,
            ),
        ],
    );
    run_ladder(
        "weight_concentration_prior_type",
        &[
            rung("prior=dp", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung(
                "prior=dd",
                20_000,
                16,
                8,
                "full",
                "kmeans",
                "dirichlet_distribution",
                100,
                1,
            ),
            // Repeated at k=32, where the stick-breaking scan and the `k`
            // `betaln`s are 4x longer — if the prior family had a measurable
            // cost anywhere, it would be here.
            rung(
                "prior=dp k=32",
                20_000,
                16,
                32,
                "full",
                "kmeans",
                DP,
                100,
                1,
            ),
            rung(
                "prior=dd k=32",
                20_000,
                16,
                32,
                "full",
                "kmeans",
                "dirichlet_distribution",
                100,
                1,
            ),
        ],
    );
    // The ladders above run to a `tol`, so part of what they time is the
    // initialization and part is the CONVERGENCE RATE. This one pins `tol = 0`
    // and a fixed `max_iter`, which makes both engines run exactly 50
    // variational iterations from a cheap random init: the only thing left in
    // the timer is the E-step and the M-step.
    run_ladder(
        "the variational loop in isolation (tol=0, max_iter=50, random init)",
        &[
            rung_fixed_iters("vb cov=full", 20_000, 16, 8, "full", "random", DP, 50),
            rung_fixed_iters("vb cov=tied", 20_000, 16, 8, "tied", "random", DP, 50),
            rung_fixed_iters("vb cov=diag", 20_000, 16, 8, "diag", "random", DP, 50),
            rung_fixed_iters(
                "vb cov=spherical",
                20_000,
                16,
                8,
                "spherical",
                "random",
                DP,
                50,
            ),
            rung_fixed_iters("vb full d=64", 20_000, 64, 8, "full", "random", DP, 50),
            rung_fixed_iters("vb tied d=64", 20_000, 64, 8, "tied", "random", DP, 50),
            rung_fixed_iters("vb full k=32", 20_000, 16, 32, "full", "random", DP, 50),
        ],
    );
    run_ladder(
        "n_init (restart multiplier)",
        &[
            rung("n_init=1", 20_000, 16, 8, "full", "kmeans", DP, 100, 1),
            rung("n_init=3", 20_000, 16, 8, "full", "kmeans", DP, 100, 3),
        ],
    );
}

/// DEVICE-vs-HOST EM engine ladder — the twin of
/// `gaussian_mixture_perf_test.rs::gmm_device_vs_host_ladder`, forcing
/// `MLRS_GMM_DEVICE` to bypass `gmm_device_applicable`'s size floor so every
/// rung is a genuine A/B. Sweeps `n_samples`, `covariance_type`,
/// `n_features`, `n_components` — the same axes, since [`GmmDevice`]'s
/// `e_step_biased` shares the whole `O(n·k·d²)` kernel set `e_step` does
/// (module docs) — plus `weight_concentration_prior_type`, which this
/// estimator alone has: its cost lives entirely in the `O(k)` bias fold
/// [`BayesianGaussianMixture::log_weight_term`] computes on HOST before every
/// device E-step launch, so this rung is the one place a widening gap between
/// `dp`/`dd` would show up as device overhead the plain model's ladder cannot
/// exercise.
///
/// Only meaningful on real cuda/rocm hardware. On cpu/wgpu this still
/// compiles and runs (both columns report the same host-engine numbers, or
/// this logs and skips via the same capability gates
/// `gaussian_mixture_perf_test.rs` checks), so the harness works unmodified
/// wherever it is run.
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn bgm_device_vs_host_ladder() {
    if mlrs_backend::capability::active_backend_name() == "cpu" {
        println!("\n=== device vs host EM engine: skipped (cpu backend, no device arm) ===");
        return;
    }
    if !mlrs_backend::capability::f64_device_kernels_available()
        || !mlrs_backend::capability::f64_transcendental_supported()
    {
        println!(
            "\n=== device vs host EM engine: skipped (backend lacks f64 device kernels or \
             f64 transcendentals — see gmm_device.rs module docs) ==="
        );
        return;
    }

    const DP: &str = "dirichlet_process";
    const DD: &str = "dirichlet_distribution";
    let rungs: Vec<Rung> = vec![
        rung_fixed_iters("n=2000", 2_000, 16, 8, "full", "random", DP, 30),
        rung_fixed_iters("n=20000", 20_000, 16, 8, "full", "random", DP, 30),
        rung_fixed_iters("n=200000", 200_000, 16, 8, "full", "random", DP, 30),
        rung_fixed_iters("cov=full n=200000", 200_000, 16, 8, "full", "random", DP, 30),
        rung_fixed_iters("cov=tied n=200000", 200_000, 16, 8, "tied", "random", DP, 30),
        rung_fixed_iters("cov=diag n=200000", 200_000, 16, 8, "diag", "random", DP, 30),
        rung_fixed_iters(
            "cov=spherical n=200000",
            200_000,
            16,
            8,
            "spherical",
            "random",
            DP,
            30,
        ),
        rung_fixed_iters("d=64 n=200000", 200_000, 64, 8, "full", "random", DP, 30),
        rung_fixed_iters("d=256 n=200000", 200_000, 256, 8, "full", "random", DP, 30),
        rung_fixed_iters("k=2 n=200000", 200_000, 16, 2, "full", "random", DP, 30),
        rung_fixed_iters("k=32 n=200000", 200_000, 16, 32, "full", "random", DP, 30),
        rung_fixed_iters("prior=dp n=200000", 200_000, 16, 8, "full", "random", DP, 30),
        rung_fixed_iters("prior=dd n=200000", 200_000, 16, 8, "full", "random", DD, 30),
    ];

    println!("\n=== BGM device vs host EM engine (MLRS_GMM_DEVICE forced) ===");
    println!(
        "{:<24} {:>12} {:>12} {:>8}",
        "rung", "host ms", "device ms", "dev/host"
    );
    println!("{}", "-".repeat(60));
    for r in &rungs {
        let x = make_blobs(r.n, r.d, r.k, 42);
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

        let (host_wall, _) = {
            let _g = abflag::force("MLRS_GMM_DEVICE", "0");
            measure(|| fit(&mut pool, &x, r))
        };
        let (dev_wall, _) = {
            let _g = abflag::force("MLRS_GMM_DEVICE", "1");
            measure(|| fit(&mut pool, &x, r))
        };
        println!(
            "{:<24} {host_wall:>12.2} {dev_wall:>12.2} {:>7.2}x",
            r.label,
            dev_wall / host_wall
        );
    }
}

/// Worker-pool width sweep, forced through the shared `MLRS_GMM_UNITS`
/// `abflag` knob so a flat sweep proves the knob is LIVE rather than proving
/// the pool does not help ([[mlrs-bench-verify-knob-is-live]]).
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn bgm_worker_pool_knee() {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let r = rung_fixed_iters(
        "full 20000x16 k=8",
        20_000,
        16,
        8,
        "full",
        "random",
        "dirichlet_process",
        50,
    );
    let x = make_blobs(r.n, r.d, r.k, 42);
    println!("\n=== worker-pool width (MLRS_GMM_UNITS) ===");
    println!(
        "{:<10} {:>10} {:>10} {:>8}",
        "units", "fit ms", "cpu ms", "speedup"
    );
    println!("{}", "-".repeat(42));
    let mut base = f64::NAN;
    for units in [1usize, 2, 4, 8, 12, 16] {
        let _g = abflag::force("MLRS_GMM_UNITS", &units.to_string());
        let (wall, cpu) = measure(|| fit(&mut pool, &x, &r));
        if units == 1 {
            base = wall;
        }
        println!("{units:<10} {wall:>10.2} {cpu:>10.2} {:>8.2}x", base / wall);
    }
}
