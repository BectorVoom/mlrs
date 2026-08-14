//! Host normal-equations formation (`prims::gram_host::centered_gram_xty`)
//! oracle validation (RIDGE-POS-PERF).
//!
//! The prim is pure host code, so it runs identically under every backend
//! feature; what varies is only which arm `Ridge::fit` routes to. The oracle is
//! a direct, deliberately naive f64 triple loop — NOT a re-derivation of the
//! blocked/transposed/threaded algorithm under test, so a bug in the tiling,
//! the `4 × 4` micro-kernel, the triangle mirroring, or the worker split cannot
//! be reproduced by the reference.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use mlrs_backend::abflag;
use mlrs_backend::prims::gram_host::{
    centered_gram_xty, gram_host_applicable, gram_host_applicable_for,
};
use mlrs_core::{assert_slice_close, F64_TOL};

/// Deterministic, reproducible pseudo-random design (splitmix64), so a failure
/// is a fixed case rather than one that depends on the run.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)`.
fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `(x, y)` for one case. Column means are deliberately far from zero (each
/// column offset by `1 + j`), so a dropped or mis-indexed mean is a LARGE
/// error, not a rounding-scale one.
fn make_case(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..n * d)
        .map(|k| uniform_pm1(&mut s) + 1.0 + (k % d) as f64)
        .collect();
    let y: Vec<f64> = (0..n).map(|_| uniform_pm1(&mut s) + 4.0).collect();
    (x, y)
}

/// The naive reference: weighted means, then an explicit `√w`-scaled centered
/// design, then a plain triple loop over it.
#[allow(clippy::type_complexity)]
fn reference(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let mut xm = vec![0.0f64; d];
    let mut ym = 0.0f64;
    if fit_intercept {
        let wsum: f64 = match sw {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };
        for i in 0..n {
            let w = sw.map(|w| w[i]).unwrap_or(1.0);
            for (a, m) in xm.iter_mut().enumerate() {
                *m += w * x[i * d + a];
            }
            ym += w * y[i];
        }
        for m in xm.iter_mut() {
            *m /= wsum;
        }
        ym /= wsum;
    }

    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];
    for i in 0..n {
        let scale = sw.map(|w| w[i].sqrt()).unwrap_or(1.0);
        let row: Vec<f64> = (0..d).map(|a| (x[i * d + a] - xm[a]) * scale).collect();
        let yc = (y[i] - ym) * scale;
        for a in 0..d {
            xty[a] += row[a] * yc;
            for b in 0..d {
                gram[a * d + b] += row[a] * row[b];
            }
        }
    }
    (xm, ym, gram, xty)
}

/// Shapes chosen to hit every branch of the blocked sweep: `d < 4` (scalar tail
/// only), `d` a multiple of 4 but not of the 64-row tile, `d` odd (both the `i`
/// and `j` tails), a single column, `n` below one `ROW_BLOCK` (64), `n`
/// straddling a block boundary, and `n` large enough that the work-proportional
/// worker split spawns more than one thread.
const SHAPES: &[(usize, usize)] = &[
    (9, 1),
    (7, 3),
    (20, 4),
    (13, 5),
    (64, 8),
    (65, 17),
    (200, 20),
    (5000, 64),
    (1000, 100),
];

/// `centered_gram_xty` vs the naive reference, unweighted, `fit_intercept`.
#[test]
fn centered_gram_xty_matches_host_ref_f64() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y) = make_case(n, d, 42 + i as u64);
        let (xm, ym, gram, xty) = centered_gram_xty::<f64>(&x, &y, n, d, None, true);
        let (exp_xm, exp_ym, exp_gram, exp_xty) = reference(&x, &y, n, d, None, true);
        assert_slice_close(&xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[ym], &[exp_ym], &F64_TOL);
        assert_slice_close(&gram, &exp_gram, &F64_TOL);
        assert_slice_close(&xty, &exp_xty, &F64_TOL);
    }
}

/// The `f32` ingress arm: the DATA is `f32`, but every accumulator is `f64`, so
/// the result must match a reference run on the same values widened to `f64`
/// (i.e. the only error allowed is the input's own representation error, which
/// is shared by both sides).
#[test]
fn centered_gram_xty_matches_host_ref_f32() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x64, y64) = make_case(n, d, 7 + i as u64);
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y64.iter().map(|&v| v as f32).collect();
        let xw: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let yw: Vec<f64> = y32.iter().map(|&v| v as f64).collect();

        let (xm, ym, gram, xty) = centered_gram_xty::<f32>(&x32, &y32, n, d, None, true);
        let (exp_xm, exp_ym, exp_gram, exp_xty) = reference(&xw, &yw, n, d, None, true);
        assert_slice_close(&xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[ym], &[exp_ym], &F64_TOL);
        assert_slice_close(&gram, &exp_gram, &F64_TOL);
        assert_slice_close(&xty, &exp_xty, &F64_TOL);
    }
}

/// `fit_intercept = false` leaves both means at zero and returns the RAW Gram
/// (sklearn's `_preprocess_data` contract) — a case where centering the data
/// anyway would be silently wrong rather than merely imprecise.
#[test]
fn centered_gram_xty_without_intercept_is_the_raw_gram() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y) = make_case(n, d, 100 + i as u64);
        let (xm, ym, gram, xty) = centered_gram_xty::<f64>(&x, &y, n, d, None, false);
        assert!(xm.iter().all(|&v| v == 0.0), "x_mean must be zero");
        assert_eq!(ym, 0.0, "y_mean must be zero");
        let (_, _, exp_gram, exp_xty) = reference(&x, &y, n, d, None, false);
        assert_slice_close(&gram, &exp_gram, &F64_TOL);
        assert_slice_close(&xty, &exp_xty, &F64_TOL);
    }
}

/// `sample_weight`: the WEIGHTED means plus the `√w` row rescale, in one pass.
/// Includes exact zeros (a weight of 0 must drop the row from BOTH the mean and
/// the Gram) and a weight vector that is not a permutation of ones.
#[test]
fn centered_gram_xty_weighted_matches_host_ref() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y) = make_case(n, d, 900 + i as u64);
        let mut s = 5150u64 + i as u64;
        let sw: Vec<f64> = (0..n)
            .map(|r| {
                if r % 7 == 3 {
                    0.0
                } else {
                    0.25 + (uniform_pm1(&mut s) + 1.0)
                }
            })
            .collect();
        let (xm, ym, gram, xty) = centered_gram_xty::<f64>(&x, &y, n, d, Some(&sw), true);
        let (exp_xm, exp_ym, exp_gram, exp_xty) = reference(&x, &y, n, d, Some(&sw), true);
        assert_slice_close(&xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[ym], &[exp_ym], &F64_TOL);
        assert_slice_close(&gram, &exp_gram, &F64_TOL);
        assert_slice_close(&xty, &exp_xty, &F64_TOL);
    }
}

/// An all-ones `sample_weight` must reproduce the unweighted result exactly —
/// the `√1 = 1` scale and the `Σw = n` mean are exact identities, so this
/// catches a weighted arm that quietly changes the arithmetic.
#[test]
fn centered_gram_xty_unit_weights_match_unweighted() {
    let (n, d) = (300usize, 12usize);
    let (x, y) = make_case(n, d, 31337);
    let ones = vec![1.0f64; n];
    let (xm_w, ym_w, gram_w, xty_w) = centered_gram_xty::<f64>(&x, &y, n, d, Some(&ones), true);
    let (xm, ym, gram, xty) = centered_gram_xty::<f64>(&x, &y, n, d, None, true);
    assert_slice_close(&xm_w, &xm, &F64_TOL);
    assert_slice_close(&[ym_w], &[ym], &F64_TOL);
    assert_slice_close(&gram_w, &gram, &F64_TOL);
    assert_slice_close(&xty_w, &xty, &F64_TOL);
}

/// The returned Gram is the FULL symmetric matrix: only the lower triangle is
/// accumulated and the upper is mirrored at the end, so a broken mirror shows
/// up here (and would otherwise reach the solver as a zero upper triangle).
#[test]
fn centered_gram_xty_output_is_symmetric() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y) = make_case(n, d, 555 + i as u64);
        let (_, _, gram, _) = centered_gram_xty::<f64>(&x, &y, n, d, None, true);
        for a in 0..d {
            for b in 0..a {
                assert_eq!(
                    gram[a * d + b],
                    gram[b * d + a],
                    "host gram not symmetric at ({a},{b}) for n={n} d={d}"
                );
            }
        }
    }
}

/// The worker split must not change the answer: forcing 1, 3 and 16 units
/// through `MLRS_CPU_UNITS` (the [`abflag`] RAII guard, never `std::env`) must
/// give bit-identical output on a shape large enough to actually be split.
///
/// `f64` addition is not associative, so this asserts EXACT equality rather
/// than closeness only where the partition is provably irrelevant — it is: each
/// worker owns a contiguous row range and the partials are summed in worker
/// order, so a differing unit count DOES reassociate. The check is therefore
/// `assert_slice_close`, and its job is to catch a lost or double-counted row
/// range, which is a gross error, not a rounding one.
#[test]
fn centered_gram_xty_is_worker_count_independent() {
    let (n, d) = (5000usize, 24usize);
    let (x, y) = make_case(n, d, 2024);
    let mut results = Vec::new();
    for units in ["1", "3", "16"] {
        let _g = abflag::force("MLRS_CPU_UNITS", units);
        results.push(centered_gram_xty::<f64>(&x, &y, n, d, None, true));
    }
    let (exp_xm, exp_ym, exp_gram, exp_xty) = reference(&x, &y, n, d, None, true);
    for (xm, ym, gram, xty) in &results {
        assert_slice_close(xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[*ym], &[exp_ym], &F64_TOL);
        assert_slice_close(gram, &exp_gram, &F64_TOL);
        assert_slice_close(xty, &exp_xty, &F64_TOL);
    }
}

/// The `MLRS_RIDGE_GRAM_HOST` A/B knob overrides the dispatch in BOTH
/// directions, whatever the backend — the property every on-target A/B run
/// depends on.
#[test]
fn gram_host_applicable_honours_the_ab_knob() {
    {
        let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "1");
        assert!(gram_host_applicable(1_000_000, 256));
    }
    {
        let _g = abflag::force("MLRS_RIDGE_GRAM_HOST", "0");
        assert!(!gram_host_applicable(4, 2));
    }
}

/// Without an override, a tiny problem takes the host arm on EVERY backend —
/// the fixed-dispatch-cost floor, which is about launch overhead and so really
/// is machine-independent.
#[test]
fn gram_host_applicable_floor_is_backend_independent() {
    let _g = abflag::clear("MLRS_RIDGE_GRAM_HOST");
    assert!(
        gram_host_applicable(1_000, 8),
        "a 1000x8 design is below the dispatch floor on every backend"
    );
    assert!(
        gram_host_applicable(1, 1),
        "a degenerate design is below the floor on every backend"
    );
}

/// ABOVE the floor the answer is no longer a constant: it is the calibrated
/// cost model's verdict (RIDGE-ARM-CAL), measured on the machine running the
/// test. What can be asserted portably is that the verdict is a DECISION rather
/// than a coin flip — stable across calls, because the rates are measured once
/// and cached — and that cpu still always answers "host".
#[test]
fn gram_host_applicable_above_the_floor_is_calibrated_and_stable() {
    let _g = abflag::clear("MLRS_RIDGE_GRAM_HOST");

    let first = gram_host_applicable(1_000_000, 256);
    for _ in 0..3 {
        assert_eq!(
            gram_host_applicable(1_000_000, 256),
            first,
            "the calibrated verdict must be cached, not re-measured per call"
        );
    }

    if mlrs_backend::capability::active_backend_name() == "cpu" {
        assert!(first, "the cpu backend has no device arm to prefer");
    }

    // The model must also be MONOTONE in the direction that matters: the host
    // arm's advantage comes from not transferring the design, so a design that
    // is wider (more bytes per multiply-add is FALSE here — wider d means more
    // arithmetic per byte, which favours the device). Assert the weaker,
    // always-true property instead: an f64 design ships twice the bytes of the
    // f32 one for identical arithmetic, so it can never be LESS
    // host-favourable.
    let f32_verdict = gram_host_applicable_for(500_000, 64, 4);
    let f64_verdict = gram_host_applicable_for(500_000, 64, 8);
    assert!(
        !f32_verdict || f64_verdict,
        "if f32 prefers the host arm, the byte-heavier f64 design must too"
    );
}
