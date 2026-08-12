//! `RidgeCV` (RIDGECV-01) — wall-clock ATTRIBUTION probe for the GCV engine.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cpu \
//!   --test ridge_cv_perf_test -- --ignored --nocapture
//! ```
//!
//! This is deliberately NOT the head-to-head. sklearn cannot be called from
//! Rust, so the library-vs-library number lives in `scripts/bench_ridge_cv.py`
//! (which times each library in its OWN PROCESS — see that file for the three
//! verdicts an interleaved harness inverted). What this probe answers is the
//! question a speedup ratio cannot: **where the time goes, and how it scales**.
//!
//! Three ladders, each isolating one term of the cost model
//! `O(n·d²) + O(d³) + O(n_alphas·n·d)`:
//!
//! 1. `alphas_ladder` — fixed design, 1 … 200 alphas. The per-alpha term is
//!    `O(n·d)`, so the curve must be nearly FLAT at small counts and then
//!    linear with a small slope. A curve that is linear from the first rung
//!    means the eigenbasis projection has leaked back into the alpha loop,
//!    which is the whole optimization.
//! 2. `shape_ladder` — fixed alpha count, growing `(n, d)`. Isolates the
//!    `O(n·d²)` Gram + projection.
//! 3. `wide_ladder` — the `n ≤ d` route, where the `O(n³)` symmetric
//!    eigendecomposition of the `n × n` Gram dominates and the design barely
//!    matters. This is the ladder that currently carries a KNOWN LOSS against
//!    sklearn (which reaches LAPACK), and it prints the eig share so the next
//!    campaign has the number rather than a hunch.
//!
//! `MLRS_RIDGECV_REPS` overrides the min-of-N count (default 5). The ladders are
//! `#[ignore]`d: they are a measurement, not a gate, and a gate on wall clock
//! would fail on a busy machine (this one measured 2× spread while an unrelated
//! `--features rocm` suite was running on the same box).
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_algos::linear::ridge_cv::{ridge_gcv, GcvMode, GcvRoute};

/// Counter-based splitmix64 — the workspace's shared deterministic stream.
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

fn make_regression(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut sx = seed;
    let x: Vec<f64> = (0..n * d).map(|_| uniform_pm1(&mut sx)).collect();
    let mut sc = seed + 1;
    let coef: Vec<f64> = (0..d).map(|_| uniform_pm1(&mut sc)).collect();
    let mut sn = seed + 2;
    let y: Vec<f64> = (0..n)
        .map(|r| {
            let mut acc = 0.5 + 0.01 * uniform_pm1(&mut sn);
            for c in 0..d {
                acc += x[r * d + c] * coef[c];
            }
            acc
        })
        .collect();
    (x, y)
}

fn reps() -> usize {
    std::env::var("MLRS_RIDGECV_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Min-of-N milliseconds after a discarded warmup. The MINIMUM, because
/// unrelated load can only ever add time.
fn measure<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps() {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn gcv(x: &[f64], y: &[f64], n: usize, d: usize, alphas: &[f64]) -> GcvRoute {
    ridge_gcv::<f64>(
        x,
        y,
        n,
        d,
        1,
        None,
        alphas,
        true,
        GcvMode::Auto,
        false,
        false,
    )
    .expect("gcv fit")
    .route
}

fn logspace(lo: f64, hi: f64, k: usize) -> Vec<f64> {
    if k == 1 {
        return vec![10f64.powf(lo)];
    }
    (0..k)
        .map(|i| 10f64.powf(lo + (hi - lo) * i as f64 / (k - 1) as f64))
        .collect()
}

#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn alphas_ladder() {
    let (n, d) = (100_000usize, 64usize);
    let (x, y) = make_regression(n, d, 42);
    println!("\nRIDGECV alphas ladder  n={n} d={d}  (per-alpha term is O(n*d))");
    let mut base = 0.0f64;
    for k in [1usize, 3, 10, 30, 100, 200] {
        let alphas = logspace(-3.0, 3.0, k);
        let ms = measure(|| {
            gcv(&x, &y, n, d, &alphas);
        });
        if k == 1 {
            base = ms;
        }
        let per_alpha = if k > 1 { (ms - base) / (k - 1) as f64 } else { 0.0 };
        println!(
            "  alphas={k:>4}: {ms:8.2} ms   fixed={base:7.2} ms   \
             marginal={per_alpha:6.3} ms/alpha"
        );
    }
}

#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn shape_ladder() {
    let alphas = logspace(-3.0, 3.0, 30);
    println!("\nRIDGECV shape ladder  30 alphas  (Gram + projection are O(n*d^2))");
    for (n, d) in [
        (10_000usize, 16usize),
        (10_000, 64),
        (100_000, 16),
        (100_000, 64),
        (200_000, 64),
        (50_000, 128),
        (20_000, 256),
        (10_000, 512),
    ] {
        let (x, y) = make_regression(n, d, 42);
        let ms = measure(|| {
            gcv(&x, &y, n, d, &alphas);
        });
        let ndd = (n as f64) * (d as f64) * (d as f64);
        println!(
            "  n={n:>7} d={d:>4}: {ms:8.2} ms   {:6.2} GFLOP/s-equivalent \
             (2*n*d^2 / t)",
            2.0 * ndd / (ms * 1e6)
        );
    }
}

#[test]
#[ignore = "wall-clock measurement, not a gate"]
fn wide_ladder() {
    let alphas = logspace(-3.0, 3.0, 30);
    println!(
        "\nRIDGECV wide ladder (n <= d, the \"gram\" route)\n  \
         the n-only column is the SERIAL sym_eig(n) share — the known loss"
    );
    // Same `n` at two very different `d`: the difference is the design-dependent
    // work, so what is LEFT at d -> 0 is the O(n^3) eigendecomposition.
    for n in [100usize, 200, 400, 800] {
        let mut prev: Option<(usize, f64)> = None;
        let mut at_small = 0.0;
        for d in [n.max(64) * 2, n.max(64) * 8] {
            let (x, y) = make_regression(n, d, 42);
            let route = gcv(&x, &y, n, d, &alphas);
            assert_eq!(route, GcvRoute::Gram, "n={n} d={d} took the wrong route");
            let ms = measure(|| {
                gcv(&x, &y, n, d, &alphas);
            });
            if let Some((d0, ms0)) = prev {
                // Two points, one line: extrapolating back to `d = 0` leaves
                // the design-INDEPENDENT part, which on this route is the
                // serial `sym_eig(n)` plus the per-alpha `O(n^2)` work.
                let slope = (ms - ms0) / (d - d0) as f64;
                at_small = ms0 - slope * d0 as f64;
            } else {
                prev = Some((d, ms));
            }
            println!("  n={n:>5} d={d:>6}: {ms:8.2} ms");
        }
        println!(
            "        -> design-independent (≈ sym_eig({n}) + per-alpha): \
             {at_small:8.2} ms"
        );
    }
}
