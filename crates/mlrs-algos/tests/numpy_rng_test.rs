//! `feature_selection::numpy_rng` — the `numpy.random.RandomState` replica,
//! pinned against numpy itself (FSEL-01).
//!
//! Every expected value below was printed by `numpy.random.RandomState` at full
//! `repr` precision. This file is the reason `mutual_info_*`'s oracle test can
//! be a statement about the ESTIMATOR rather than about two different noise
//! streams: if the stream diverges, these tests fail here, with the divergence
//! localised to the generator instead of showing up as a 1e-3 mutual-information
//! discrepancy fifteen call frames away.
//!
//! Pure host scalar code, no backend involvement, so no capability gate.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use mlrs_algos::feature_selection::numpy_rng::NumpyRandomState;

/// `np.random.RandomState(42).randint(0, 2**32, size=5)` — the raw tempered
/// 32-bit stream, before any float packing. Pinned first because a failure here
/// means the twist or the seeding is wrong, which every other test would inherit.
#[test]
fn u32_stream_matches_numpy() {
    let mut rng = NumpyRandomState::new(42);
    let want: [u32; 5] = [
        1_608_637_542,
        3_421_126_067,
        4_083_286_876,
        787_846_414,
        3_143_890_026,
    ];
    for (i, &w) in want.iter().enumerate() {
        assert_eq!(rng.next_u32(), w, "next_u32 at draw {i}");
    }
}

/// `np.random.RandomState(42).random_sample(4)` — the `((a>>5)·2²⁶ + (b>>6))/2⁵³`
/// packing. A replica using `u64 as f64 / 2⁶⁴` passes the test above and fails
/// this one, which is exactly why both exist.
#[test]
fn double_stream_matches_numpy() {
    let mut rng = NumpyRandomState::new(42);
    let want = [
        0.374_540_118_847_362_5,
        0.950_714_306_409_916_2,
        0.731_993_941_811_405_1,
        0.598_658_484_197_036_6,
    ];
    for (i, &w) in want.iter().enumerate() {
        let got = rng.next_f64();
        assert!(
            (got - w).abs() <= 1e-15,
            "next_f64 at draw {i}: got={got:.17} want={w:.17}"
        );
    }
}

/// `np.random.RandomState(seed).standard_normal(n)` for three seeds — the polar
/// method AND its cached-pair ORDER.
///
/// The order is what the even-indexed values test: a replica that returns `f·x1`
/// first and caches `f·x2` produces the same MULTISET but transposes every
/// consecutive pair, so it agrees on `n = 1` and fails from the second value on.
#[test]
fn standard_normal_matches_numpy() {
    let cases: [(u64, &[f64]); 3] = [
        (
            42,
            &[
                0.496_714_153_011_232_7,
                -0.138_264_301_171_184_66,
                0.647_688_538_100_692_5,
                1.523_029_856_408_025_4,
                -0.234_153_374_723_335_97,
                -0.234_136_956_949_180_55,
            ],
        ),
        (
            0,
            &[
                1.764_052_345_967_664,
                0.400_157_208_367_223_3,
                0.978_737_984_105_739_2,
                2.240_893_199_201_458,
            ],
        ),
        (
            12345,
            &[
                -0.204_707_659_484_712_95,
                0.478_943_338_057_548_24,
                -0.519_438_715_056_738_1,
                -0.555_730_304_347_49,
            ],
        ),
    ];
    for (seed, want) in cases {
        let mut rng = NumpyRandomState::new(seed);
        for (i, &w) in want.iter().enumerate() {
            let got = rng.standard_normal();
            assert!(
                (got - w).abs() <= 1e-14 * w.abs().max(1.0),
                "standard_normal(seed={seed}) at draw {i}: got={got:.17} want={w:.17}"
            );
        }
    }
}

/// `standard_normal_vec` is the same stream as repeated `standard_normal`, so a
/// caller drawing an `(n, d)` block gets the C-order fill numpy gives.
#[test]
fn standard_normal_vec_matches_scalar_stream() {
    let mut a = NumpyRandomState::new(7);
    let block = a.standard_normal_vec(11);
    let mut b = NumpyRandomState::new(7);
    let scalars: Vec<f64> = (0..11).map(|_| b.standard_normal()).collect();
    assert_eq!(block, scalars);
}
