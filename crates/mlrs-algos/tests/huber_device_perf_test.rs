//! `HuberRegressor` (HUBER-02) DEVICE performance probes — the GPU twin of
//! `huber_perf_test.rs`.
//!
//! These are PROBES, not gates: they print tables and assert only what would
//! make a number meaningless (a solve that did no work, a fixture that stopped
//! exercising the outlier branch, or an A/B knob that turned out to be dead).
//! Absolute timings belong to the machine, so nothing here compares against a
//! pinned constant.
//!
//! What each table is for:
//!
//! - [`device_engine_vs_roundtrip_ladder`] — the HUBER-02 headline, A/B'd on
//!   ONE build through `MLRS_HUBER_DEVICE`. The round-trip arm pays two
//!   `n`-length transfers and two pipeline stalls per objective evaluation; the
//!   resident-`g` engine pays one `d_aug + 5` readback. The ladder walks `n` at
//!   fixed `d` and `d` at fixed `n` so the two costs can be told apart.
//! - [`device_ingress_avoids_the_upload`] — the zero-copy half. Because the
//!   synthetic intercept column is no longer materialized, a device-resident
//!   design is BORROWED, so a `fit` on data already on the device uploads
//!   nothing. Measured against the host-slice ingress, which must upload once.
//! - [`parameter_cost_sweep_device`] — every ctor parameter that measurably
//!   moves the cost, on the device arm. Same split as the cpu probe: `tol` /
//!   `max_iter` / `warm_start` move the ITERATION COUNT, `sample_weight` /
//!   `fit_intercept` move the per-evaluation cost, `epsilon` / `alpha` move the
//!   conditioning and are reported rather than asserted.
//!
//! Everything runs at `f32`. That is the one dtype every device backend here
//! supports: rocm's `cubek-matmul` rejects `f64` operands and cuda does not
//! advertise `f64` at all ([[mlrs-rocm-hardware-env]],
//! [[mlrs-cubecl-cuda-f64-not-advertised]]), so an `f64` probe would self-skip
//! on exactly the hardware being measured.
//!
//! Run with `--nocapture` to see the tables:
//! ```text
//! cargo test -p mlrs-algos --features rocm --release \
//!     --test huber_device_perf_test -- --nocapture
//! ```
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

#![cfg(not(feature = "cpu"))]

use std::time::Instant;

use mlrs_algos::linear::huber::HuberRegressor;
use mlrs_algos::typestate::Fit;
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The ladder. Walks `n` at fixed `d` (isolating the `O(n)` traffic the
/// round-trip arm pays and the resident engine does not) and `d` at fixed `n`
/// (isolating the two GEMM passes, which BOTH arms pay identically) — so a rung
/// where the A/B ratio collapses says "this one is GEMM-bound", not "the engine
/// stopped working".
const LADDER: &[(usize, usize)] = &[
    (1_000, 8),
    (10_000, 8),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (50_000, 128),
];

/// Deterministic `[-1, 1)` stream (splitmix64) so a rung is reproducible.
fn uniform_pm1(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// A design with GROSS OUTLIERS in `y` — without them every sample sits in the
/// quadratic core, the fit degenerates to least squares, and the probe measures
/// a problem no one would reach for this estimator to solve.
///
/// Byte-for-byte the `huber_perf_test.rs` generator, then narrowed to `f32`, so
/// a device rung and a cpu rung of the same `(n, d, seed)` are the SAME problem
/// and their times are comparable.
fn design(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let x = uniform_pm1(seed, n * d);
    let w = uniform_pm1(seed ^ 0xF00D, d);
    let noise = uniform_pm1(seed ^ 0xBEEF, n);
    let shock = uniform_pm1(seed ^ 0xC0DE, n);
    let y: Vec<f32> = (0..n)
        .map(|r| {
            let mut m = 1.5;
            for j in 0..d {
                m += x[r * d + j] * w[j];
            }
            // ~8 % of rows take a large additive shock.
            let gross = if shock[r] > 0.84 {
                25.0 * shock[r]
            } else {
                0.0
            };
            (m + 0.4 * noise[r] + gross) as f32
        })
        .collect();
    (x.iter().map(|&v| v as f32).collect(), y)
}

/// One host-ingress fit, timed. Returns `(seconds, n_iter_, n_outliers)`.
#[allow(clippy::too_many_arguments)]
fn timed_fit(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f32],
    y: &[f32],
    (n, d): (usize, usize),
    epsilon: f64,
    alpha: f64,
    tol: f64,
    max_iter: usize,
    fit_intercept: bool,
    sw: Option<&[f32]>,
    seed_params: Option<Vec<f64>>,
) -> (f64, usize, usize, Vec<f64>) {
    let mut b = HuberRegressor::<f32>::builder()
        .epsilon(epsilon)
        .alpha(alpha)
        .tol(tol)
        .max_iter(max_iter)
        .fit_intercept(fit_intercept);
    if let Some(seed) = seed_params {
        b = b.warm_start(true).init_params(seed);
    }
    let est = b.build::<f32>().expect("huber build");
    let t0 = Instant::now();
    let fitted = est
        .fit_from_host_slice(pool, x, y, (n, d), sw)
        .expect("huber fit");
    let secs = t0.elapsed().as_secs_f64();
    (
        secs,
        fitted.n_iter(),
        fitted.outliers().iter().filter(|&&o| o).count(),
        fitted.warm_start_params().to_vec(),
    )
}

/// Min-of-`reps` after a warm-up, so a rung reports the machine's best rather
/// than whatever else the box was doing ([[mlrs-cpu-bench-separate-processes]]:
/// a loaded box has INVERTED a verdict in this repo before, and the same is
/// true of a GPU shared with a compositor).
fn best_of<Fn>(reps: usize, mut f: Fn) -> (f64, usize, usize)
where
    Fn: FnMut() -> (f64, usize, usize),
{
    let _ = f();
    let mut best = (f64::INFINITY, 0usize, 0usize);
    for _ in 0..reps {
        let r = f();
        if r.0 < best.0 {
            best = r;
        }
    }
    best
}

/// Reps per arm in the A/B probes, and the rule for how they are scheduled.
///
/// Five rather than the cpu probe's three because these run against a shared
/// integrated GPU. And the arms ALTERNATE — arm A, arm B, arm A, … — rather
/// than running in blocks, which is the trap: a load burst shorter than a whole
/// block taxes exactly one arm and shows up as a clean, plausible, WRONG ratio.
/// Alternating means any burst short of the whole probe hits every arm, and the
/// min-of-`AB_REPS` then discards it from all of them. This repo has had a
/// verdict inverted by exactly this ([[mlrs-cpu-bench-separate-processes]]), on
/// a box no busier than this one.
///
/// Each rep re-forces its own knob through [`abflag`], which is thread-local, so
/// the alternation cannot leak one arm's setting into another's timing.
const AB_REPS: usize = 5;

/// The 1-minute load average, printed next to every table.
///
/// Not decoration. A number produced at load 270 on a 16-core box (which this
/// machine has genuinely been during this work) is not a measurement, and the
/// only way a reader can tell is if the table says so.
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "?".into())
}

/// THE ladder that sets `prims::huber_objective::HUBER_DEVICE_MIN_WORK`: the
/// fused host pass against the device engine, A/B'd through
/// `MLRS_HUBER_ENGINE` on ONE build.
///
/// A Huber fit is an L-BFGS solve — a few dozen objective evaluations, each of
/// which must synchronize once because the driver needs the loss and gradient
/// on the host before choosing its next step. So the device arm carries a FIXED
/// per-evaluation floor (one stall + four launches) that does not shrink with
/// the problem, while the host arm's cost is proportional to `n·d` from the
/// first row. There is therefore a crossover, and the only honest way to place
/// it is to measure it.
///
/// Reported, never asserted against a constant: the crossover is a property of
/// the machine, and on an integrated GPU sharing DRAM with the host it may not
/// exist at all within any reasonable `n·d`. That is a legitimate outcome and
/// the table says so rather than a threshold pretending otherwise.
///
/// Read the printed `loadavg` before reading the numbers. Interleaving and
/// min-of-N help, but a saturated machine has INVERTED a verdict in this repo
/// before ([[mlrs-cpu-bench-separate-processes]]), and an integrated GPU is
/// contended by anything using the CPU's memory bandwidth, not just by other
/// GPU work.
#[test]
fn host_vs_device_crossover() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    println!(
        "\n[huber engine crossover] backend={} f32  loadavg={}",
        capability::active_backend_name(),
        loadavg()
    );
    println!(
        "{:>8} {:>5} {:>11} | {:>10} {:>12} {:>10} | {:>6}",
        "n", "d", "n·d", "host(ms)", "device(ms)", "dev/host", "iter"
    );

    for &(n, d) in LADDER {
        let (x, y) = design(n, d, 7);
        // One rep of one arm. Kept as a single `FnMut` selected by a flag so the
        // two arms can ALTERNATE while sharing the `&mut pool` neither closure
        // could hold across the other's call.
        let one = |engine: &'static str, pool: &mut BufferPool<ActiveRuntime>| {
            let _g = abflag::force("MLRS_HUBER_ENGINE", engine);
            let (s, it, out, _) = timed_fit(
                pool,
                &x,
                &y,
                (n, d),
                1.35,
                1e-4,
                1e-5,
                100,
                true,
                None,
                None,
            );
            (s, it, out)
        };
        let mut host = (f64::INFINITY, 0usize, 0usize);
        let mut device = (f64::INFINITY, 0usize, 0usize);
        // Warm BOTH before timing either: page faults on a fresh design, the
        // worker-pool spawn, and every device kernel's JIT.
        let _ = one("host", &mut pool);
        let _ = one("device", &mut pool);
        for _ in 0..AB_REPS {
            let h = one("host", &mut pool);
            if h.0 < host.0 {
                host = h;
            }
            let dv = one("device", &mut pool);
            if dv.0 < device.0 {
                device = dv;
            }
        }
        println!(
            "{n:>8} {d:>5} {:>11} | {:>10.3} {:>12.3} {:>9.2}x | {:>6}",
            n * d,
            host.0 * 1e3,
            device.0 * 1e3,
            device.0 / host.0,
            host.1,
        );
        assert!(
            host.2 > 0 && device.2 > 0,
            "n={n} d={d}: an arm classified no sample as an outlier, so the fit \
             degenerated to least squares and the rung is not measuring Huber"
        );
        let slack = (n / 200).max(2);
        assert!(
            host.2.abs_diff(device.2) <= slack,
            "n={n} d={d}: the two ENGINES disagree on the outlier count \
             ({} vs {}, slack {slack}) — they are not solving the same problem, \
             so the ratio above is meaningless",
            host.2,
            device.2
        );
    }
}

/// The HUBER-02 headline: the resident-`g` engine against the round-trip arm it
/// replaced, A/B'd on ONE build.
///
/// Both arms run the SAME two GEMMs over the design; they differ only in what
/// crosses the bus between them. So the expected shape is a large ratio where
/// `n` dominates and a shrinking one where `d` does — and the table prints
/// `µs/iter` next to the wall clock precisely so a rung that moved because the
/// SOLVER took a different number of steps cannot be read as an engine win.
#[test]
fn device_engine_vs_roundtrip_ladder() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    println!(
        "\n[huber device ladder] backend={} f32",
        capability::active_backend_name()
    );
    println!(
        "{:>8} {:>5} | {:>12} {:>12} {:>8} | {:>12} {:>8} | {:>6} {:>6}",
        "n", "d", "resident(ms)", "roundtrip(ms)", "vs rt", "gemm(ms)", "vs gemm", "iter", "outl"
    );

    let mut any_win = false;
    for &(n, d) in LADDER {
        let (x, y) = design(n, d, 7);

        // One rep of one arm — the three arms then ALTERNATE, so a load burst
        // shorter than the whole probe hits all of them rather than taxing
        // whichever happened to run during it (see `AB_REPS`).
        let one = |knob: Option<&'static str>, pool: &mut BufferPool<ActiveRuntime>| {
            let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
            let _g = match knob {
                Some(v) => abflag::force("MLRS_HUBER_DEVICE", v),
                None => abflag::clear("MLRS_HUBER_DEVICE"),
            };
            let (s, it, out, _) = timed_fit(
                pool,
                &x,
                &y,
                (n, d),
                1.35,
                1e-4,
                1e-5,
                100,
                true,
                None,
                None,
            );
            (s, it, out)
        };
        const ARMS: [Option<&str>; 3] = [None, Some("0"), Some("gemm")];
        let mut best = [(f64::INFINITY, 0usize, 0usize); 3];
        for arm in ARMS {
            let _ = one(arm, &mut pool);
        }
        for _ in 0..AB_REPS {
            for (i, arm) in ARMS.iter().enumerate() {
                let r = one(*arm, &mut pool);
                if r.0 < best[i].0 {
                    best[i] = r;
                }
            }
        }
        let (resident, roundtrip, gemm) = (best[0], best[1], best[2]);

        println!(
            "{n:>8} {d:>5} | {:>12.3} {:>12.3} {:>7.2}x | {:>12.3} {:>7.1}x | {:>6} {:>6}",
            resident.0 * 1e3,
            roundtrip.0 * 1e3,
            roundtrip.0 / resident.0,
            gemm.0 * 1e3,
            gemm.0 / resident.0,
            resident.1,
            resident.2,
        );
        any_win |= roundtrip.0 / resident.0 > 1.0;

        assert!(
            resident.1 > 1,
            "n={n} d={d}: the solve converged in {} iteration(s), so the rung \
             measures setup rather than the objective evaluation",
            resident.1
        );
        assert!(
            resident.2 > 0,
            "n={n} d={d}: no sample was classified as an outlier, so the fit \
             degenerated to least squares and the probe is not measuring Huber"
        );
        // The two arms must still land on the SAME fit — an engine that won by
        // converging somewhere else is not a win.
        //
        // Compared with a SLACK rather than for equality: the arms reduce in
        // different widths and orders, so at `f32` they stop at points a few
        // units in the last place apart, and a residual sitting within that of
        // the `ε·σ` threshold flips category. One or two rows out of `n` is
        // that effect; a real divergence moves the count by percent.
        let slack = (n / 200).max(2);
        assert!(
            resident.2.abs_diff(roundtrip.2) <= slack,
            "n={n} d={d}: the two arms disagree on the outlier count \
             ({} vs {}, slack {slack}), so the ratio above is comparing \
             different fits",
            resident.2,
            roundtrip.2
        );
    }
    // Not a per-rung gate (a small-`n`, large-`d` rung is GEMM-bound and may
    // legitimately tie), but if NO rung improves then the knob is dead or the
    // engine is not reached, and the whole table is vacuous.
    assert!(
        any_win,
        "the resident-g engine did not beat the round-trip arm on a SINGLE \
         rung — MLRS_HUBER_DEVICE is not reaching the engine \
         (see mlrs-bench-verify-knob-is-live)"
    );
}

/// The zero-copy ingress: a design already on the device is BORROWED, so the
/// fit uploads nothing.
///
/// Before HUBER-02 the device arm materialized the augmented `n × (d+1)`
/// operand, which forced a device-resident design through `to_host` → host
/// augment → `from_host` — three passes over `n·d` and a sync, to write a
/// column of ones. The host-slice ingress still pays ONE upload (it has to),
/// so the gap between the two columns is what the borrow buys.
#[test]
fn device_ingress_avoids_the_upload() {
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    println!(
        "\n[huber ingress] backend={} f32",
        capability::active_backend_name()
    );
    println!(
        "{:>8} {:>5} | {:>12} {:>12} {:>8}",
        "n", "d", "device(ms)", "host(ms)", "ratio"
    );

    for &(n, d) in &[(100_000usize, 16usize), (50_000, 128)] {
        let (x, y) = design(n, d, 7);
        let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

        let dev = best_of(3, || {
            let est = HuberRegressor::<f32>::builder()
                .build::<f32>()
                .expect("huber build");
            let t0 = Instant::now();
            let fitted = est
                .fit(&mut pool, &xd, Some(&yd), (n, d))
                .expect("huber device fit");
            (
                t0.elapsed().as_secs_f64(),
                fitted.n_iter(),
                fitted.outliers().iter().filter(|&&o| o).count(),
            )
        });
        let host = best_of(3, || {
            let (s, it, out, _) = timed_fit(
                &mut pool,
                &x,
                &y,
                (n, d),
                1.35,
                1e-4,
                1e-5,
                100,
                true,
                None,
                None,
            );
            (s, it, out)
        });

        println!(
            "{n:>8} {d:>5} | {:>12.3} {:>12.3} {:>7.2}x",
            dev.0 * 1e3,
            host.0 * 1e3,
            host.0 / dev.0
        );
        assert_eq!(
            dev.2, host.2,
            "n={n} d={d}: the two ingresses classified a different number of \
             outliers ({} vs {}), so they are not the same fit",
            dev.2, host.2
        );
        xd.release_into(&mut pool);
        yd.release_into(&mut pool);
    }
}

/// Every ctor parameter that measurably changes the COST, on the DEVICE arm.
///
/// Two mechanisms, kept separate the way the cpu probe keeps them:
///
/// - **Iteration count.** `tol` and `max_iter` change where the solve stops and
///   `warm_start` starts it closer. All three are ASSERTED, because the
///   direction is structural rather than a property of this fixture.
/// - **Cost per evaluation.** `sample_weight` makes the classify kernel index a
///   real weight vector instead of taking its `weighted == 0` path;
///   `fit_intercept` adds the `bias` scalar and the `Σgᵢ` fold quantity — which
///   on the device arm, unlike the cpu one, does NOT widen the GEMM operand,
///   since the synthetic column is never materialized. Read `µs/iter` for both.
///
/// `epsilon` and `alpha` are REPORTED, not asserted: they change the
/// conditioning of the objective, and which way that moves the iteration count
/// is a property of the DESIGN, not of the parameter.
#[test]
fn parameter_cost_sweep_device() {
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (n, d) = (100_000, 16);
    let (x, y) = design(n, d, 11);
    let sw: Vec<f32> = uniform_pm1(99, n)
        .iter()
        .map(|v| (1.5 + v) as f32)
        .collect();

    println!(
        "\n[huber device parameter cost] backend={} n={n} d={d} f32",
        capability::active_backend_name()
    );
    println!(
        "{:>26} | {:>10} {:>10} {:>7}",
        "configuration", "fit (ms)", "µs/iter", "n_iter"
    );

    let row = |label: &str,
               pool: &mut BufferPool<ActiveRuntime>,
               eps: f64,
               alpha: f64,
               tol: f64,
               max_iter: usize,
               fi: bool,
               weighted: bool,
               seed: Option<Vec<f64>>| {
        let swr = weighted.then_some(sw.as_slice());
        let (secs, iters, _) = best_of(3, || {
            let (s, it, out, _) = timed_fit(
                pool,
                &x,
                &y,
                (n, d),
                eps,
                alpha,
                tol,
                max_iter,
                fi,
                swr,
                seed.clone(),
            );
            (s, it, out)
        });
        println!(
            "{label:>26} | {:>10.3} {:>10.1} {iters:>7}",
            secs * 1e3,
            secs * 1e6 / iters.max(1) as f64
        );
        (secs, iters)
    };

    let (_, base_iters) = row(
        "default", &mut pool, 1.35, 1e-4, 1e-5, 100, true, false, None,
    );
    // --- epsilon: the conditioning knob (reported). ----------------------- #
    row(
        "epsilon=1.05",
        &mut pool,
        1.05,
        1e-4,
        1e-5,
        100,
        true,
        false,
        None,
    );
    row(
        "epsilon=10.0",
        &mut pool,
        10.0,
        1e-4,
        1e-5,
        100,
        true,
        false,
        None,
    );
    // --- alpha: regularization (reported). -------------------------------- #
    row(
        "alpha=0", &mut pool, 1.35, 0.0, 1e-5, 100, true, false, None,
    );
    row(
        "alpha=100",
        &mut pool,
        1.35,
        100.0,
        1e-5,
        100,
        true,
        false,
        None,
    );
    // --- tol / max_iter: the stop (asserted). ----------------------------- #
    // `tol` is scikit-learn's `gtol` — a bound on the LARGEST gradient entry,
    // which for this objective is an `O(Σ swᵢ)` sum, i.e. `O(n)`. At
    // `n = 100 000` a "loose" tolerance therefore has to be `O(10⁴)`, not the
    // `O(1)` the cpu probe gets away with at `f64`: an `f32` solve simply never
    // drives the gradient down to single digits, so `tol = 5` is still tighter
    // than where it stops and would leave the iteration count unmoved — an
    // assertion that fails for a reason that has nothing to do with `tol`.
    let (_, loose_iters) = row(
        "tol=1e4", &mut pool, 1.35, 1e-4, 1e4, 100, true, false, None,
    );
    let (_, cap_iters) = row(
        "max_iter=5",
        &mut pool,
        1.35,
        1e-4,
        1e-5,
        5,
        true,
        false,
        None,
    );
    // --- per-evaluation cost (reported). ---------------------------------- #
    row(
        "fit_intercept=False",
        &mut pool,
        1.35,
        1e-4,
        1e-5,
        100,
        false,
        false,
        None,
    );
    row(
        "sample_weight",
        &mut pool,
        1.35,
        1e-4,
        1e-5,
        100,
        true,
        true,
        None,
    );
    // --- warm_start: sklearn's real use case, a capped fit continued. ----- #
    let (_, _, _, partial) = timed_fit(
        &mut pool,
        &x,
        &y,
        (n, d),
        1.35,
        1e-4,
        1e-5,
        5,
        true,
        None,
        None,
    );
    let (_, warm_iters) = row(
        "warm_start (partial seed)",
        &mut pool,
        1.35,
        1e-4,
        1e-5,
        100,
        true,
        false,
        Some(partial),
    );

    assert!(
        loose_iters < base_iters,
        "a tol of 1e4 took {loose_iters} iterations against the default's \
         {base_iters} — `tol` is not reaching the gradient stop"
    );
    assert_eq!(
        cap_iters, 5,
        "max_iter=5 reported {cap_iters} iterations, so the cap is not the cap"
    );
    assert!(
        warm_iters < base_iters,
        "warm-starting from a 5-iteration partial fit took {warm_iters} \
         iterations against a cold start's {base_iters} — the seed is not \
         being consumed"
    );
}
