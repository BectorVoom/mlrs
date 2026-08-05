//! `GaussianMixture` (MIX-01) wall-clock probe — mlrs against sklearn.
//!
//! The Rust half of a two-process comparison whose other half is
//! `scripts/bench_gmm.py`. Both build the SAME design from the same
//! counter-based splitmix64 stream, fit with the same hyperparameters, and
//! print the same columns, so the two tables can be divided rung by rung.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test gaussian_mixture_perf_test -- --ignored --nocapture
//! python3 scripts/bench_gmm.py
//! ```
//!
//! ## What the ladders are for
//! Each one sweeps a hyperparameter that changes the COMPLEXITY CLASS, not just
//! the constant — those are the parameters worth a measurement:
//!
//! | ladder | axis | why it matters |
//! |---|---|---|
//! | `cov=*` | `covariance_type` | `full`/`tied` are `O(n·k·d²)`, `diag`/`spherical` `O(n·k·d)`. The biggest lever, and where `tied`'s hoisted E-step shows up. |
//! | `d=*` | `n_features` | the quadratic axis; the triangular Mahalanobis win grows with it |
//! | `k=*` | `n_components` | linear, but it multiplies the quadratic term |
//! | `n=*` | `n_samples` | linear; also where the worker pool starts paying |
//! | `init=*` | `init_params` | `kmeans` runs a whole Lloyd fit before EM; the random routes run none but need more EM |
//! | `n_init=*` | restart count | a pure multiplier — checks the restart loop leaks no per-restart setup |
//!
//! ## Methodology
//! min-of-N after a discarded warmup, with BOTH wall and process-CPU time
//! reported. Two prior campaigns had a verdict inverted by a co-tenant job on
//! the same box ([[mlrs-cpu-bench-separate-processes]],
//! [[mlrs-bench-verify-knob-is-live]]), so the CPU column is not decoration: if
//! one engine's `cpu/wall` is far above 1 and the other's is near 1, the
//! wall-clock ratio is reporting thread count rather than efficiency.
//!
//! `MLRS_GMM_REPS` overrides the repeat count; `MLRS_GMM_UNITS` forces the
//! worker-pool width (the `abflag` knob `gmm_host` reads), which is what the
//! `units=*` ladder sweeps.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::mixture::gaussian_mixture::GaussianMixture;
use mlrs_algos::typestate::Fitted;
use mlrs_backend::abflag;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64, byte-identical to `scripts/bench_gmm.py`.
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

/// `k` well-separated isotropic blobs — the same array `bench_gmm.py` builds.
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
/// Not an `#[ignore]`d probe — a plain test, because a silent drift between
/// `make_blobs` here and `scripts/bench_gmm.py`'s would turn every published
/// ratio into a comparison of two different problems, and nothing else in the
/// suite would notice. The reference values were printed by the Python side.
#[test]
fn shared_design_matches_the_python_probe() {
    let got = make_blobs(4, 3, 2, 42);
    let want = [
        5.376918011619, -7.882205097444, -6.206825485956, -4.261804099267,
        -8.978946090085, 5.235874905861, 4.016352812844, -6.537024689807,
        -5.165802603336, -3.509310159428, -9.068499958212, 7.245708178618,
    ];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-9,
            "make_blobs diverged from scripts/bench_gmm.py at {i}: {g} vs {w}"
        );
    }
}

fn reps() -> usize {
    std::env::var("MLRS_GMM_REPS")
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
/// Python's `process_time()` has an mlrs-specific trap — cubecl spins threads,
/// so it charges idle spin as work ([[mlrs-bench-verify-knob-is-live]]). Here
/// there is no cubecl in the loop at all (the EM engine is pure host code), so
/// the reading is meaningful; it is still reported alongside wall rather than
/// instead of it.
fn cpu_time_ms() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return f64::NAN;
    };
    // The `comm` field is parenthesized and may itself contain spaces, so the
    // only safe split point is the LAST ')' — everything after it is the
    // fixed-position field list starting at `state` (field 3).
    let Some((_, tail)) = stat.rsplit_once(')') else {
        return f64::NAN;
    };
    let fields: Vec<&str> = tail.split_whitespace().collect();
    // Overall field 14 is utime and 15 is stime; `tail` starts at field 3, so
    // they are indices 11 and 12 here.
    let utime: f64 = fields.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = fields.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    // USER_HZ is 100 on every Linux target this repo builds for.
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
    max_iter: usize,
    n_init: usize,
    /// `0.0` forces the loop to run exactly `max_iter` iterations in BOTH
    /// engines, which is what isolates the EM loop from the initialization.
    tol: f64,
}

fn rung(
    label: impl Into<String>,
    n: usize,
    d: usize,
    k: usize,
    cov: &'static str,
    init: &'static str,
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
        max_iter,
        n_init,
        tol: 1e-3,
    }
}

/// A rung that runs a FIXED number of EM iterations (`tol = 0`).
#[allow(clippy::too_many_arguments)]
fn rung_fixed_iters(
    label: impl Into<String>,
    n: usize,
    d: usize,
    k: usize,
    cov: &'static str,
    init: &'static str,
    max_iter: usize,
) -> Rung {
    Rung {
        tol: 0.0,
        ..rung(label, n, d, k, cov, init, max_iter, 1)
    }
}

fn fit(pool: &mut BufferPool<ActiveRuntime>, x: &[f64], r: &Rung) -> GaussianMixture<f64, Fitted> {
    GaussianMixture::<f64>::builder()
        .n_components(r.k)
        .covariance_type(r.cov)
        .init_params(r.init)
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
        "{:<24} {:>10} {:>10} {:>10} {:>10} {:>7} {:>6}",
        "rung", "fit ms", "cpu ms", "pred ms", "score ms", "n_iter", "units"
    );
    println!("{}", "-".repeat(82));
    for r in rungs {
        let x = make_blobs(r.n, r.d, r.k, 42);
        let (fit_wall, fit_cpu) = measure(|| fit(&mut pool, &x, r));
        let fitted = fit(&mut pool, &x, r);
        let (pred_wall, _) = measure(|| fitted.predict_labels_host(&x, (r.n, r.d)).unwrap());
        let (score_wall, _) = measure(|| fitted.score_samples_host(&x, (r.n, r.d)).unwrap());
        let units = abflag::var("MLRS_GMM_UNITS").unwrap_or_else(|| "auto".to_string());
        println!(
            "{:<24} {fit_wall:>10.2} {fit_cpu:>10.2} {pred_wall:>10.2} {score_wall:>10.2} {:>7} {units:>6}",
            r.label,
            fitted.n_iter()
        );
    }
}

/// The full comparison ladder. `--ignored --nocapture`, release only.
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn gmm_perf_ladders() {
    run_ladder(
        "covariance_type (the complexity-class lever)",
        &[
            rung("cov=full", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("cov=tied", 20_000, 16, 8, "tied", "kmeans", 100, 1),
            rung("cov=diag", 20_000, 16, 8, "diag", "kmeans", 100, 1),
            rung("cov=spherical", 20_000, 16, 8, "spherical", "kmeans", 100, 1),
        ],
    );
    run_ladder(
        "n_features (the quadratic axis)",
        &[
            rung("d=4", 20_000, 4, 8, "full", "kmeans", 100, 1),
            rung("d=16", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("d=64", 20_000, 64, 8, "full", "kmeans", 100, 1),
            rung("d=128", 20_000, 128, 8, "full", "kmeans", 100, 1),
        ],
    );
    run_ladder(
        "n_components",
        &[
            rung("k=2", 20_000, 16, 2, "full", "kmeans", 100, 1),
            rung("k=8", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("k=32", 20_000, 16, 32, "full", "kmeans", 100, 1),
        ],
    );
    run_ladder(
        "n_samples",
        &[
            rung("n=2000", 2_000, 16, 8, "full", "kmeans", 100, 1),
            rung("n=20000", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("n=200000", 200_000, 16, 8, "full", "kmeans", 100, 1),
        ],
    );
    run_ladder(
        "init_params",
        &[
            rung("init=kmeans", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("init=k-means++", 20_000, 16, 8, "full", "k-means++", 100, 1),
            rung("init=random", 20_000, 16, 8, "full", "random", 100, 1),
            rung(
                "init=random_from_data",
                20_000,
                16,
                8,
                "full",
                "random_from_data",
                100,
                1,
            ),
        ],
    );
    // The ladders above run to sklearn's default `tol`, which on a separable
    // design converges in 2-3 iterations — so most of what they time is the
    // INITIALIZATION, not the EM loop. This one pins `tol = 0` and a fixed
    // `max_iter`, which makes BOTH engines run exactly 50 EM iterations from a
    // cheap random init: the only thing left in the timer is the E-step and
    // M-step themselves.
    run_ladder(
        "EM loop in isolation (tol=0, max_iter=50, random init)",
        &[
            rung_fixed_iters("em cov=full", 20_000, 16, 8, "full", "random", 50),
            rung_fixed_iters("em cov=tied", 20_000, 16, 8, "tied", "random", 50),
            rung_fixed_iters("em cov=diag", 20_000, 16, 8, "diag", "random", 50),
            rung_fixed_iters("em cov=spherical", 20_000, 16, 8, "spherical", "random", 50),
            rung_fixed_iters("em full d=64", 20_000, 64, 8, "full", "random", 50),
            rung_fixed_iters("em tied d=64", 20_000, 64, 8, "tied", "random", 50),
        ],
    );
    run_ladder(
        "n_init (restart multiplier)",
        &[
            rung("n_init=1", 20_000, 16, 8, "full", "kmeans", 100, 1),
            rung("n_init=3", 20_000, 16, 8, "full", "kmeans", 100, 3),
        ],
    );
}

/// Worker-pool width sweep — the knee of the host parallelization, forced
/// through the `MLRS_GMM_UNITS` `abflag` knob so a flat sweep proves the knob is
/// LIVE rather than proving the pool does not help
/// ([[mlrs-bench-verify-knob-is-live]]).
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn gmm_worker_pool_knee() {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let r = rung("full 20000x16 k=8", 20_000, 16, 8, "full", "kmeans", 100, 1);
    let x = make_blobs(r.n, r.d, r.k, 42);
    println!("\n=== worker-pool width (MLRS_GMM_UNITS) ===");
    println!("{:<10} {:>10} {:>10} {:>8}", "units", "fit ms", "cpu ms", "speedup");
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

/// DEVICE-vs-HOST EM engine ladder (`MLRS_GMM_DEVICE` forces either arm,
/// bypassing `gmm_device_applicable`'s size floor so every rung below is a
/// genuine A/B rather than a report of the gate's own threshold). Sweeps the
/// three axes the design doc calls out as mattering: `n_samples` (device
/// should win at large `n`, lose at small `n` — the whole point of the
/// size-gated predicate), `covariance_type` (`full`/`tied` are the expensive
/// `O(n·k·d²)` forms), and `n_features`/`n_components`.
///
/// Only meaningful on real cuda/rocm hardware — see `gmm_device.rs`'s module
/// docs for why the device arm is gated off wgpu at `f64` (no transcendentals)
/// and off cpu always. On cpu/wgpu this still compiles and RUNS (both columns
/// report the same host-engine numbers, or the device-forced column errors out
/// via the hard capability gates inside `gmm_device_applicable` — in which case
/// this test logs and skips rather than asserting a ratio), so the harness
/// exists and works unmodified when the user runs it on Kaggle/T4 hardware.
/// Per this repo's own convention (`ridge_default_perf_test.rs` et al.) this
/// prints a table; it never asserts a specific speedup ratio.
#[test]
#[ignore = "wall-clock probe; run with --release --ignored --nocapture"]
fn gmm_device_vs_host_ladder() {
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

    let rungs: Vec<Rung> = vec![
        rung_fixed_iters("n=2000", 2_000, 16, 8, "full", "random", 30),
        rung_fixed_iters("n=20000", 20_000, 16, 8, "full", "random", 30),
        rung_fixed_iters("n=200000", 200_000, 16, 8, "full", "random", 30),
        rung_fixed_iters("cov=full n=200000", 200_000, 16, 8, "full", "random", 30),
        rung_fixed_iters("cov=tied n=200000", 200_000, 16, 8, "tied", "random", 30),
        rung_fixed_iters("cov=diag n=200000", 200_000, 16, 8, "diag", "random", 30),
        rung_fixed_iters(
            "cov=spherical n=200000",
            200_000,
            16,
            8,
            "spherical",
            "random",
            30,
        ),
        rung_fixed_iters("d=64 n=200000", 200_000, 64, 8, "full", "random", 30),
        rung_fixed_iters("d=256 n=200000", 200_000, 256, 8, "full", "random", 30),
        rung_fixed_iters("k=2 n=200000", 200_000, 16, 2, "full", "random", 30),
        rung_fixed_iters("k=32 n=200000", 200_000, 16, 32, "full", "random", 30),
    ];

    println!("\n=== device vs host EM engine (MLRS_GMM_DEVICE forced) ===");
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
