//! `HuberRegressor` (HUBER-01) cpu performance probes.
//!
//! These are PROBES, not gates: they print a table and assert only the property
//! that would make the number meaningless (a solve that did no work, or a knob
//! that turned out to be dead). Absolute timings belong to the machine, so
//! nothing here compares against a pinned constant — `scripts/bench_huber.py` is
//! where the mlrs-vs-scikit-learn verdict is produced, in separate processes on
//! a quiet box (the `mlrs-cpu-bench-separate-processes` finding).
//!
//! What this file is for is the part a Python benchmark cannot see:
//!
//! - [`fit_cost_is_evaluations_times_pass`] — the ladder, printed as BOTH
//!   wall-clock and `n_iter_`, so a rung that got slower can be attributed to
//!   "the pass got slower" or "the solver took more steps" without guessing.
//!   Those two call for opposite fixes.
//! - [`parameter_cost_sweep`] — every ctor parameter that changes the ITERATION
//!   COUNT (and so the cost), measured rather than assumed: `epsilon` and
//!   `alpha` change the conditioning, `tol` and `max_iter` change the stop,
//!   `warm_start` removes the early iterations from a refit, `sample_weight`
//!   switches the row loop to its weighted monomorphization, and
//!   `fit_intercept` adds one column to the augmented width.
//! - [`worker_knee_sweep`] — `MLRS_HUBER_ELEMS_PER_UNIT` A/B'd through
//!   `mlrs_backend::abflag` (never `std::env::set_var` — that is an environ data
//!   race, see the `mlrs-abflag-test-knobs` note), so the knee inherited from
//!   `svm_objective` is re-derived on this pass rather than assumed to transfer.
//!
//! Run with `--nocapture` to see the tables:
//! ```text
//! cargo test -p mlrs-algos --features cpu --release --test huber_perf_test -- --nocapture
//! ```
//! A debug build measures `-O0` Rust and is not informative — the fused host
//! pass exists because it is compiled at `-O3` (see the prim's module docs), so
//! the `--release` flag is not optional for reading these numbers.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::time::Instant;

use mlrs_algos::linear::huber::HuberRegressor;
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The ladder. Walks `n` at fixed `d` (isolating the streaming cost of the
/// design, which is what the fused pass and the worker split address) and `d` at
/// fixed `n` (isolating the per-row dot product and the `d`-length gradient
/// accumulate).
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
fn design(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = uniform_pm1(seed, n * d);
    let w = uniform_pm1(seed ^ 0xF00D, d);
    let noise = uniform_pm1(seed ^ 0xBEEF, n);
    let shock = uniform_pm1(seed ^ 0xC0DE, n);
    let y = (0..n)
        .map(|r| {
            let mut m = 1.5;
            for j in 0..d {
                m += x[r * d + j] * w[j];
            }
            // ~8 % of rows take a large additive shock.
            let gross = if shock[r] > 0.84 { 25.0 * shock[r] } else { 0.0 };
            m + 0.4 * noise[r] + gross
        })
        .collect();
    (x, y)
}

/// One fit, timed. Returns `(seconds, n_iter_, n_outliers)`.
#[allow(clippy::too_many_arguments)]
fn timed_fit(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    y: &[f64],
    (n, d): (usize, usize),
    epsilon: f64,
    alpha: f64,
    tol: f64,
    max_iter: usize,
    fit_intercept: bool,
    sw: Option<&[f64]>,
    seed_params: Option<Vec<f64>>,
) -> (f64, usize, usize, Vec<f64>) {
    let mut b = HuberRegressor::<f64>::builder()
        .epsilon(epsilon)
        .alpha(alpha)
        .tol(tol)
        .max_iter(max_iter)
        .fit_intercept(fit_intercept);
    if let Some(seed) = seed_params {
        b = b.warm_start(true).init_params(seed);
    }
    let est = b.build::<f64>().expect("huber build");
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

/// The ladder, reported as wall-clock AND iteration count.
///
/// The cost of a Huber fit is `n_iter × (one pass over the design)` plus the
/// line-search evaluations, so a rung is only comparable to another when its
/// iteration count is known — that is why the count is printed next to the time
/// rather than left implicit.
#[test]
fn fit_cost_is_evaluations_times_pass() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // The probe prints `evals=` per fit — the count `n_iter` hides. Forced
    // through `abflag` (thread-local) rather than the environment.
    let _probe = abflag::force("MLRS_HUBER_PROBE", "1");
    println!(
        "\n[huber ladder] backend={}  units={}",
        capability::active_backend_name(),
        capability::cpu_launch_units()
    );
    println!(
        "{:>8} {:>5} | {:>10} {:>10} {:>7} {:>9}",
        "n", "d", "fit (ms)", "µs/iter", "n_iter", "outliers"
    );
    for &(n, d) in LADDER {
        let (x, y) = design(n, d, 7);
        // One warm-up fit: the first touch of a fresh design pays its page
        // faults, and the worker pool's threads are spawned per solve.
        let _ = timed_fit(
            &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 100, true, None, None,
        );
        let mut best = f64::INFINITY;
        let mut iters = 0;
        let mut outliers = 0;
        for _ in 0..3 {
            let (secs, it, out, _) = timed_fit(
                &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 100, true, None, None,
            );
            if secs < best {
                best = secs;
                iters = it;
                outliers = out;
            }
        }
        println!(
            "{n:>8} {d:>5} | {:>10.3} {:>10.1} {iters:>7} {outliers:>9}",
            best * 1e3,
            best * 1e6 / iters.max(1) as f64,
        );
        assert!(
            iters > 1,
            "n={n} d={d}: the solve converged in {iters} iteration(s), so the \
             rung is measuring setup rather than the objective pass"
        );
        assert!(
            outliers > 0,
            "n={n} d={d}: no sample was classified as an outlier, so the fit \
             degenerated to least squares and the probe is not measuring Huber"
        );
    }
}

/// Every ctor parameter that measurably changes the COST, measured.
///
/// Two distinct mechanisms show up here and the table separates them:
///
/// - **Iteration count.** `tol` and `max_iter` change where the solve stops, and
///   `warm_start` starts it closer — all three move the cost by moving `n_iter`,
///   and all three are ASSERTED below because the direction is structural.
/// - **Cost per pass.** `sample_weight` switches the row loop to its `WEIGHTED`
///   monomorphization, which reads one extra `f64` per row; `fit_intercept` adds
///   one entry to the augmented weight vector. These move the cost at a roughly
///   fixed iteration count, so `µs/iter` is the column to read for them.
///
/// `epsilon` and `alpha` are REPORTED, not asserted. They change the
/// conditioning of the objective, and which way that moves the iteration count
/// is a property of the DESIGN, not of the parameter: on this fixture
/// `epsilon = 1.05` converges in FEWER steps than the default while costing more
/// per step, which is the opposite of what a "tighter epsilon is harder" story
/// would predict. An assertion here would be pinning one dataset's conditioning
/// and would fire on an unrelated change to the generator.
#[test]
fn parameter_cost_sweep() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (n, d) = (100_000, 16);
    let (x, y) = design(n, d, 11);
    let sw: Vec<f64> = uniform_pm1(99, n).iter().map(|v| 1.5 + v).collect();

    // The converged default fit, and the parameters it warm-starts from.
    let (_, _, _, converged_params) = timed_fit(
        &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 100, true, None, None,
    );

    println!("\n[huber parameter cost] n={n} d={d} f64");
    println!(
        "{:>26} | {:>10} {:>10} {:>7}",
        "configuration", "fit (ms)", "µs/iter", "n_iter"
    );

    let mut row = |label: &str,
                   pool: &mut BufferPool<ActiveRuntime>,
                   eps: f64,
                   alpha: f64,
                   tol: f64,
                   max_iter: usize,
                   fi: bool,
                   weighted: bool,
                   seed: Option<Vec<f64>>| {
        let swr = weighted.then_some(sw.as_slice());
        let mut best = f64::INFINITY;
        let mut iters = 0;
        for _ in 0..3 {
            let (secs, it, _, _) = timed_fit(
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
            if secs < best {
                best = secs;
                iters = it;
            }
        }
        println!(
            "{label:>26} | {:>10.3} {:>10.1} {iters:>7}",
            best * 1e3,
            best * 1e6 / iters.max(1) as f64
        );
        (best, iters)
    };

    let (_, base_iters) = row(
        "default", &mut pool, 1.35, 1e-4, 1e-5, 100, true, false, None,
    );
    // --- epsilon: the conditioning knob. --------------------------------- #
    let (_, eps105_iters) = row(
        "epsilon=1.05", &mut pool, 1.05, 1e-4, 1e-5, 100, true, false, None,
    );
    row(
        "epsilon=10.0", &mut pool, 10.0, 1e-4, 1e-5, 100, true, false, None,
    );
    // --- alpha: regularization. ------------------------------------------ #
    row("alpha=0", &mut pool, 1.35, 0.0, 1e-5, 100, true, false, None);
    let (_, alpha100_iters) = row(
        "alpha=100", &mut pool, 1.35, 100.0, 1e-5, 100, true, false, None,
    );
    // --- tol / max_iter: the stop. --------------------------------------- #
    let (_, loose_iters) = row(
        "tol=5.0", &mut pool, 1.35, 1e-4, 5.0, 100, true, false, None,
    );
    let (_, cap_iters) = row(
        "max_iter=5", &mut pool, 1.35, 1e-4, 1e-5, 5, true, false, None,
    );
    // --- per-pass cost, at whatever iteration count they land on. -------- #
    row(
        "fit_intercept=False", &mut pool, 1.35, 1e-4, 1e-5, 100, false, false, None,
    );
    row(
        "sample_weight", &mut pool, 1.35, 1e-4, 1e-5, 100, true, true, None,
    );
    // --- warm_start: BOTH the real use case and the degenerate one. ------ #
    //
    // The real one is sklearn's: a capped fit continued by another capped fit.
    // Seeding a capped solve from a partial one reaches a better point for the
    // same budget, which is the whole reason the parameter exists.
    let (_, _, _, partial_params) = timed_fit(
        &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 5, true, None, None,
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
        Some(partial_params),
    );
    // The degenerate one is worth its own row because it is a TRAP, not a
    // saving: seeding from an ALREADY-CONVERGED fit leaves the gradient below
    // the `ftol` plateau but still above `gtol` (mlrs stops on the former), so
    // the strong-Wolfe search spends its whole `maxls = 50` budget failing to
    // find a decreasing step and the "free" refit costs MORE than a cold one.
    // sklearn has the same shape of behaviour for the same reason; neither is a
    // sensible thing to ask for, and the row exists so nobody has to rediscover
    // that from a profile.
    row(
        "warm_start (converged seed)",
        &mut pool,
        1.35,
        1e-4,
        1e-5,
        100,
        true,
        false,
        Some(converged_params),
    );

    // The properties that make the table meaningful. Anything that is merely
    // FASTER is reported, not asserted — that is the machine's business.
    assert!(
        cap_iters == 5,
        "max_iter=5 ran {cap_iters} iterations — the cap is not being applied, \
         so its row is not measuring what it claims"
    );
    assert!(
        loose_iters < base_iters,
        "tol=5.0 ({loose_iters}) did not stop earlier than the default \
         ({base_iters}), so the `tol` row is measuring nothing"
    );
    assert!(
        warm_iters < base_iters,
        "a warm start from a PARTIAL fit still took {warm_iters} iterations \
         against the cold {base_iters} — the seed is not reaching the solver, \
         and this row would silently report a non-existent saving"
    );
    // `epsilon` / `alpha` are reported, not asserted (doc comment above). What
    // IS asserted about them is only that they did not silently stop mattering:
    // a conditioning knob that changes neither the iteration count nor the
    // per-pass cost on any of its settings would mean the row is measuring the
    // default three times over.
    assert!(
        eps105_iters != base_iters || alpha100_iters != base_iters,
        "neither epsilon=1.05 ({eps105_iters}) nor alpha=100 ({alpha100_iters}) \
         moved the iteration count off the default ({base_iters}) — the \
         conditioning rows are measuring nothing"
    );
}

/// Re-derive `MLRS_HUBER_ELEMS_PER_UNIT` — the design elements one worker must
/// be handed before splitting the fused pass pays — on THIS pass.
///
/// The constant is inherited from `svm_objective`'s measured curve, which is the
/// right prior (same shape of pass, same persistent pool) but is not evidence
/// about this one: the Huber row loop carries three extra scalar reductions and
/// a branch the SVM loop does not, so its per-row cost — and therefore the point
/// where a barrier crossing stops being worth it — could sit elsewhere.
///
/// The override goes through `mlrs_backend::abflag`, never `std::env::set_var`:
/// mutating the environ from a test is a data race against every other thread in
/// the process, and the guard's thread-local override is what makes the sweep
/// both safe and actually EFFECTIVE (a stale value silently flattens the whole
/// sweep — see the `mlrs-bench-verify-knob-is-live` note).
#[test]
fn worker_knee_sweep() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    println!(
        "\n[huber knee sweep] MLRS_HUBER_ELEMS_PER_UNIT  (units available: {})",
        capability::cpu_launch_units()
    );
    let knees = [1usize << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16];
    print!("{:>8} {:>5} |", "n", "d");
    for k in knees {
        print!(" {:>9}", format!("1<<{}", k.trailing_zeros()));
    }
    println!();

    let mut any_spread = false;
    for &(n, d) in &[(1_000usize, 16usize), (10_000, 16), (100_000, 16), (50_000, 64)] {
        let (x, y) = design(n, d, 13);
        print!("{n:>8} {d:>5} |");
        let mut times = Vec::new();
        for knee in knees {
            let _guard = abflag::force("MLRS_HUBER_ELEMS_PER_UNIT", &knee.to_string());
            // Warm-up, then min-of-3: the pool is spawned per solve, so the
            // first fit of a configuration pays for thread creation.
            let _ = timed_fit(
                &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 100, true, None, None,
            );
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let (secs, _, _, _) = timed_fit(
                    &mut pool, &x, &y, (n, d), 1.35, 1e-4, 1e-5, 100, true, None, None,
                );
                best = best.min(secs);
            }
            print!(" {:>9.3}", best * 1e3);
            times.push(best);
        }
        println!();
        let lo = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = times.iter().cloned().fold(0.0f64, f64::max);
        if hi / lo > 1.2 {
            any_spread = true;
        }
    }
    // A COMPLETELY flat sweep across a 32x range of knees on the largest rung
    // would mean the knob never reached the pass — the exact failure mode the
    // `mlrs-bench-verify-knob-is-live` note records, where a stale build made a
    // whole sweep vacuous. The knee genuinely changes the worker count at
    // 100 000 x 16 (1 600 000 elements: 1<<11 asks for 781 workers and clamps to
    // the machine's, 1<<16 asks for 24), so SOME row must move.
    assert!(
        any_spread,
        "the knee sweep is flat on every rung — MLRS_HUBER_ELEMS_PER_UNIT is not \
         reaching `host_units`, so this table is vacuous"
    );
}
