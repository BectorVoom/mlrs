//! Phase-17 RandomForest feasibility spike — A3 per-tree **COST** benchmark
//! (TREE-01, Plan 04). This is the one genuine unknown of the spike: is the
//! `O(samples × bins)` single-owner GATHER build tractable on a realistic load?
//! A3 cannot be reasoned away — only measured (RESEARCH §A3, the single
//! MEDIUM-confidence row). The Plan-05 verdict cites THESE printed numbers, never
//! "it compiled" (Pitfall 5).
//!
//! What this probe records (run targeted, `--nocapture`):
//!   - A full depth-8 per-tree build on the representative ≈1000 samples × 20
//!     features load at BOTH `n_bins = 64` AND `n_bins = 128` (D-05 / D-10),
//!     printing wall-clock per build.
//!   - The 64-vs-128 wall-clock DELTA, so the D-06 "fewer bins" ADJUST lever is
//!     data-backed rather than asserted.
//!   - A samples-scaling sweep (250 / 500 / 1000 at fixed 128 bins) printing the
//!     time ratio vs the sample ratio, so the cost SHAPE (sub-quadratic vs
//!     pathological) is observable — D-05 judges the shape, NOT an absolute
//!     wall-clock ceiling.
//!   - A frontier-memory observation: whether histogram scratch is bounded by the
//!     active frontier or by the cumulative node count (Pitfall 6 → the D-06
//!     "frontier-only" lever).
//!
//! It is a plain `std::time::Instant` wall-clock probe — NOT a Criterion
//! micro-benchmark (RESEARCH §Benchmark Harness). It MUST be run targeted
//! (`--test tree_bench`) and must NOT pull the full mlrs-backend cpu suite
//! (project memory: `backend-test-suite-slow`, `full-cargo-test-exhausts-disk`).
//! Zero new packages — `std::time::Instant` + the existing `tree_spike` kernels.
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests`.

mod tree_spike;

use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::capability;
use std::time::{Duration, Instant};
use tree_spike::{build_tree, from_f64};

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic host RNG (splitmix64) — no `rand` dependency, so the spike adds
// zero packages (T-17-SC). Seeded, so every run benchmarks the identical dataset.
// ─────────────────────────────────────────────────────────────────────────────

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic `f64` in `[0, 1)` (53-bit mantissa).
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Representative dataset generation: a binary-classification load with a genuine
// diagonal decision boundary over the first three features, so the axis-aligned
// binned tree must staircase-approximate it and actually GROWS to depth (a
// non-trivial tree — the timing must be of a real build, not a no-op).
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(binned [n_samples*n_feat, values in 0..n_bins], y [{0,1}],
/// bin_edges [n_feat][n_bins-1])`. Bin edges are host-precomputed per-feature
/// quantiles (D-10 — NO on-device sort).
fn gen_dataset<F>(
    n_samples: usize,
    n_feat: usize,
    n_bins: usize,
    seed: u64,
) -> (Vec<u32>, Vec<F>, Vec<Vec<f64>>)
where
    F: bytemuck::Pod,
{
    let mut st = seed;
    let mut raw = vec![0.0f64; n_samples * n_feat];
    for v in raw.iter_mut() {
        *v = next_unit(&mut st);
    }

    // Diagonal boundary over features 0,1,2 → forces a deep staircase tree.
    let mut y = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let score = raw[s * n_feat] + raw[s * n_feat + 1] + raw[s * n_feat + 2];
        let label = if score > 1.5 { 1.0 } else { 0.0 };
        y.push(from_f64::<F>(label));
    }

    // Per-feature quantile bin edges (n_bins-1 thresholds each). Build each inner
    // Vec independently so the `with_capacity` reservation survives — `vec![proto;
    // n]` would CLONE the prototype, and `Vec::clone` copies only len (0), dropping
    // the reservation for every element (IN-02).
    let mut bin_edges: Vec<Vec<f64>> =
        (0..n_feat).map(|_| Vec::with_capacity(n_bins - 1)).collect();
    for f in 0..n_feat {
        let mut col: Vec<f64> = (0..n_samples).map(|s| raw[s * n_feat + f]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in host RNG output"));
        for i in 1..n_bins {
            let idx = (i * n_samples / n_bins).min(n_samples - 1);
            bin_edges[f].push(col[idx]);
        }
    }

    // Digitize each raw value into 0..n_bins using its feature's edges.
    let mut binned = vec![0u32; n_samples * n_feat];
    for s in 0..n_samples {
        for f in 0..n_feat {
            let v = raw[s * n_feat + f];
            let edges = &bin_edges[f];
            let mut b = 0u32;
            while (b as usize) < edges.len() && v > edges[b as usize] {
                b += 1;
            }
            binned[s * n_feat + f] = b;
        }
    }

    (binned, y, bin_edges)
}

// ─────────────────────────────────────────────────────────────────────────────
// Timed driver — wraps the host per-level `build_tree` loop (which drives the
// three Plan-02 device kernels) in `std::time::Instant`.
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn timed_build<F>(
    label: &str,
    binned: &[u32],
    y: &[F],
    bin_edges: &[Vec<f64>],
    n_samples: usize,
    n_feat: usize,
    n_bins: usize,
    max_depth: usize,
    min_samples: usize,
) -> (Duration, usize)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let t = Instant::now();
    let (nodes, leaves) = build_tree::<F>(
        binned, y, bin_edges, n_samples, n_feat, n_bins, max_depth, min_samples,
    );
    let dt = t.elapsed();
    let internal = nodes.iter().filter(|n| n.colid >= 0).count();
    println!(
        "  [{label:<8}] n={n_samples:>4} feat={n_feat} bins={n_bins:>3} depth={max_depth}: \
         {dt:>10.3?}  ({} nodes, {} internal, {} leaves)",
        nodes.len(),
        internal,
        leaves.len()
    );
    (dt, nodes.len())
}

// ─────────────────────────────────────────────────────────────────────────────
// The benchmark body. `full_sweep` is true for the always-on f32 path and false
// for the gated f64 headline (keeps total wall-clock bounded — D-05 judges shape,
// and the shape is dtype-independent).
// ─────────────────────────────────────────────────────────────────────────────

fn run_bench<F>(tag: &str, full_sweep: bool)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n_feat = 20usize;
    let max_depth = 8usize;
    let min_samples = 2usize;
    // The headline / sweep bin count, named once so the `peak_cells` memory note
    // below can't silently misreport if the headline bin count ever changes (IN-01).
    const HEADLINE_BINS: usize = 128;

    println!("\n=== A3 per-tree COST benchmark [{tag}] (feat={n_feat}, depth={max_depth}) ===");

    // Headline: the representative ≈1000×20×depth-8 load at 128 AND 64 bins.
    let (b128, y128, e128) = gen_dataset::<F>(1000, n_feat, HEADLINE_BINS, 0xA3C0_u64);
    let (t128, nodes_128) = timed_build::<F>(
        "headline", &b128, &y128, &e128, 1000, n_feat, HEADLINE_BINS, max_depth, min_samples,
    );

    let (b64, y64, e64) = gen_dataset::<F>(1000, n_feat, 64, 0xA3C0_u64);
    let (t64, nodes_64) = timed_build::<F>(
        "headline", &b64, &y64, &e64, 1000, n_feat, 64, max_depth, min_samples,
    );

    // 64-vs-128 DELTA — the D-06 "fewer bins" lever, data-backed.
    println!(
        "  64-vs-128 delta @ n=1000: 64bins={t64:.3?}  128bins={t128:.3?}  \
         delta(128-64)={:.3?}  ratio={:.2}x",
        t128.saturating_sub(t64),
        t128.as_secs_f64() / t64.as_secs_f64().max(1e-9)
    );

    // Samples-scaling sweep at fixed 128 bins — judge the cost SHAPE (D-05). The
    // 1000-point reuses the headline timing (no redundant heavy build).
    if full_sweep {
        println!("  -- sample-scaling sweep @ 128 bins (cost SHAPE, not a ceiling) --");
        let mut points: Vec<(usize, f64)> = Vec::new();
        for &n in &[250usize, 500usize] {
            let (bn, yn, en) = gen_dataset::<F>(n, n_feat, HEADLINE_BINS, 0xA3C0_u64);
            let (dt, _) = timed_build::<F>(
                "sweep-n", &bn, &yn, &en, n, n_feat, HEADLINE_BINS, max_depth, min_samples,
            );
            points.push((n, dt.as_secs_f64()));
        }
        points.push((1000, t128.as_secs_f64()));

        let mut prev: Option<(usize, f64)> = None;
        for (n, t) in &points {
            if let Some((pn, pt)) = prev {
                let sample_ratio = *n as f64 / pn as f64;
                let time_ratio = t / pt.max(1e-9);
                println!(
                    "       {pn}->{n}: samples x{sample_ratio:.1}, time x{time_ratio:.2}  \
                     (sub-quadratic iff time-ratio <= samples-ratio^2 = {:.1})",
                    sample_ratio * sample_ratio
                );
            }
            prev = Some((*n, *t));
        }
    }

    // Correctness sanity (NON-no-op): the timed builds produced real, grown trees.
    // Without this, a silent no-op build would report a meaningless fast time.
    assert!(
        nodes_128 > 3,
        "headline 128-bin build must produce a non-trivial tree (got {nodes_128} nodes) — \
         the timing must be of a REAL build, not a no-op"
    );
    assert!(
        nodes_64 > 3,
        "headline 64-bin build must produce a non-trivial tree (got {nodes_64} nodes)"
    );

    // Frontier-memory observation (Pitfall 6 → D-06 "frontier-only" lever). The
    // spike's `build_tree` launches the histogram sized by `nodes.len()` (the
    // CUMULATIVE node count so far), not the active frontier — so the deepest
    // level's scratch is `final_nodes × n_feat × n_bins × 2` (count + vsum).
    let peak_cells = nodes_128 * n_feat * HEADLINE_BINS;
    println!(
        "  frontier-memory note: peak histogram scratch ≈ {peak_cells} cells × 2 buffers \
         (count+vsum) = {nodes_128} nodes × {n_feat} feat × {HEADLINE_BINS} bins."
    );
    println!(
        "    build_tree sizes the histogram by CUMULATIVE node count (nodes.len()), NOT the \
         active frontier — so the D-06 'frontier-only' lever is a genuine future optimization, \
         not already realized (Pitfall 6)."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point: f32 always runs (the always-available path); f64 is the cpu
// correctness gate and SKIPS-with-log on an adapter lacking f64 (e.g. rocm).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tree_bench_per_tree_cost() {
    let _ = env_logger::builder().is_test(true).try_init();

    // f32: always run, with the full samples-scaling sweep.
    run_bench::<f32>("f32", true);

    // f64: the cpu correctness gate. Headline-only (64 + 128) to bound wall-clock.
    if capability::skip_f64_with_log() {
        println!(
            "tree_bench: f64 path SKIPPED on {} (no f64 support on this adapter)",
            capability::active_backend_name()
        );
    } else {
        run_bench::<f64>("f64", false);
    }
}
