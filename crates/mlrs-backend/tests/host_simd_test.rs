//! `host_simd` gate: the AVX2 twins must be BIT-IDENTICAL to the baseline
//! bodies they dispatch around.
//!
//! Every `*_host` prim is compiled for the x86-64 baseline and pairs its hot
//! function with a `#[target_feature(avx2, fma)]` twin (see
//! `prims::host_simd`). The whole argument for applying that to numerical code
//! without re-validating each prim's output is that it CANNOT change a value:
//! widening a register does not reassociate independent accumulators, and Rust
//! does not contract `a + b*c` into an FMA without an explicit `mul_add`.
//!
//! This file is that argument as a test, per prim, comparing raw BITS rather
//! than a tolerance — a tolerance would pass on a genuine reassociation, which
//! is exactly the failure mode being excluded.
//!
//! `knn_host_test::avx2_and_baseline_agree_bitwise` does the same for the k-NN
//! scan (it needs that file's fixtures).
//!
//! ## Every test here asserts the knob is LIVE first
//! `avx2_available()` caches the CPUID and environment halves of its answer, so
//! an A/B that only flipped a cached value would compare a body against ITSELF
//! and pass unconditionally. `assert_knob_live` fails loudly instead of
//! quietly proving nothing — and it is what the split in
//! `host_simd::avx2_available` exists for.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)]` module.

use mlrs_backend::abflag;
use mlrs_backend::prims::gram_host::{centered_gram_multi_xty, centered_gram_xty};
use mlrs_backend::prims::host_simd::avx2_available;
use mlrs_backend::prims::linear_predict::linear_predict_host;

/// Counter-based splitmix64 (the workspace bench/probe generator).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn design(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0)
        .collect()
}

/// Skip-with-reason unless this machine can actually run BOTH bodies.
///
/// Returns `false` on a CPU without AVX2, where the twin and the baseline are
/// the same code and the comparison below would be vacuous — but a vacuum that
/// is a property of the HARDWARE, not of a stale cached flag.
fn assert_knob_live() -> bool {
    // Thread-local, never `set_var`: libtest runs these on parallel threads.
    let forced_off = {
        let _g = abflag::force("MLRS_HOST_AVX2", "0");
        avx2_available()
    };
    let default_on = {
        let _g = abflag::clear("MLRS_HOST_AVX2");
        avx2_available()
    };
    assert!(
        !forced_off,
        "MLRS_HOST_AVX2=0 did not disable the AVX2 body — the knob is cached \
         somewhere and every A/B below would compare a body against itself"
    );
    if !default_on {
        eprintln!("skipping: this CPU reports no AVX2/FMA, so there is one body, not two");
    }
    default_on
}

/// Run `f` with the AVX2 twin forced on, then with it forced off.
fn both_ways<R>(f: impl Fn() -> R) -> (R, R) {
    let wide = {
        let _g = abflag::clear("MLRS_HOST_AVX2");
        f()
    };
    let base = {
        let _g = abflag::force("MLRS_HOST_AVX2", "0");
        f()
    };
    (wide, base)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gram_host_sweep_agrees_bitwise() {
    if !assert_knob_live() {
        return;
    }
    // Both a `d` that fills whole 4x4 register blocks and one with a tail, since
    // the twin covers the tail's scalar pair sweep too.
    for (n, d) in [(4_000usize, 16usize), (2_500, 23)] {
        let x = design(n * d, 7);
        let y = design(n, 11);
        for intercept in [true, false] {
            let (wide, base) =
                both_ways(|| centered_gram_xty::<f32>(&x, &y, n, d, None, intercept));
            assert_eq!(bits(&wide.0), bits(&base.0), "gram n={n} d={d}");
            assert_eq!(wide.1.to_bits(), base.1.to_bits(), "y_mean n={n} d={d}");
            assert_eq!(bits(&wide.2), bits(&base.2), "x_mean n={n} d={d}");
            assert_eq!(bits(&wide.3), bits(&base.3), "xty n={n} d={d}");
        }
        // Sample weights take the `√w` scaling branch of the same sweep.
        let sw: Vec<f64> = (0..n).map(|i| 0.25 + (i % 7) as f64).collect();
        let (wide, base) = both_ways(|| centered_gram_xty::<f32>(&x, &y, n, d, Some(&sw), true));
        assert_eq!(bits(&wide.0), bits(&base.0), "weighted gram n={n} d={d}");
        assert_eq!(bits(&wide.3), bits(&base.3), "weighted xty n={n} d={d}");
    }
}

#[test]
fn gram_host_multi_sweep_agrees_bitwise() {
    if !assert_knob_live() {
        return;
    }
    let (n, d, k) = (3_000usize, 17usize, 3usize);
    let x = design(n * d, 21);
    let y = design(n * k, 22);
    let (wide, base) = both_ways(|| centered_gram_multi_xty::<f32>(&x, &y, n, d, k, None, true));
    assert_eq!(bits(&wide.0), bits(&base.0), "multi gram");
    assert_eq!(bits(&wide.1), bits(&base.1), "multi y_mean");
    assert_eq!(bits(&wide.2), bits(&base.2), "multi x_mean");
    assert_eq!(bits(&wide.3), bits(&base.3), "multi xty");
}

#[test]
fn linear_predict_host_agrees_bitwise() {
    if !assert_knob_live() {
        return;
    }
    // A feature count past the lane split and one below it, so both the chunked
    // body and the scalar remainder are covered.
    for (m, n) in [(5_000usize, 32usize), (3_000, 7)] {
        let x = design(m * n, 3);
        let coef = design(n, 5);
        let (wide, base) =
            both_ways(|| linear_predict_host::<f32>(&x, &coef, 0.25f32, (m, n)).expect("predict"));
        assert_eq!(
            wide.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            base.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "predict m={m} n={n}"
        );
        assert_eq!(wide.operand_finite, base.operand_finite);
    }
}

#[test]
fn linear_predict_host_agrees_bitwise_f64() {
    if !assert_knob_live() {
        return;
    }
    // The twin is monomorphized per float width, so f32 coverage does not imply
    // f64 coverage — and f64 is where the register width matters most (SSE2
    // gives it TWO lanes).
    let (m, n) = (4_000usize, 24usize);
    let x: Vec<f64> = design(m * n, 31).iter().map(|&v| v as f64).collect();
    let coef: Vec<f64> = design(n, 33).iter().map(|&v| v as f64).collect();
    let (wide, base) =
        both_ways(|| linear_predict_host::<f64>(&x, &coef, -0.5f64, (m, n)).expect("predict"));
    assert_eq!(bits(&wide.values), bits(&base.values));
}
