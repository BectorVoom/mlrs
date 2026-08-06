//! numpy legacy-`RandomState` parity gate (MODSEL-RS-01).
//!
//! Every randomized splitter in this surface is only as faithful as the
//! generator underneath it, so this file pins the generator itself against
//! values produced by a live numpy — before any splitter logic is in the
//! picture. A drift here would show up as a *plausible but wrong* split
//! everywhere else, which is exactly the failure mode that is hardest to spot
//! downstream.
//!
//! The expected values were produced with numpy 2.4 and are reproducible with:
//!
//! ```text
//! python -c "import numpy as np; rs = np.random.RandomState(42); print(rs.permutation(10))"
//! ```
//!
//! (The MT19937 stream is a fixed standard, not a numpy version detail — these
//! literals hold for every numpy that still ships the legacy `RandomState`.)

use mlrs_algos::model_selection::rng::{
    approximate_mode, sample_without_replacement, NumpyRandomState, SampleMethod,
};

#[test]
fn next_u32_matches_numpy_seed_42() {
    // np.random.RandomState(42).randint(0, 2**32, size=5, dtype=np.uint32)
    let mut rng = NumpyRandomState::from_seed(42);
    let got: Vec<u32> = (0..5).map(|_| rng.next_u32()).collect();
    assert_eq!(
        got,
        vec![1608637542, 3421126067, 4083286876, 787846414, 3143890026]
    );
}

#[test]
fn random_sample_matches_numpy() {
    let mut rng = NumpyRandomState::from_seed(42);
    let got: Vec<f64> = (0..3).map(|_| rng.random_sample()).collect();
    let want = [0.3745401188473625, 0.9507143064099162, 0.7319939418114051];
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            g, w,
            "random_sample must be bit-identical, not merely close"
        );
    }
}

#[test]
fn permutation_matches_numpy() {
    let mut rng = NumpyRandomState::from_seed(42);
    assert_eq!(rng.permutation(10), vec![8, 1, 5, 0, 7, 2, 9, 4, 3, 6]);

    let mut rng = NumpyRandomState::from_seed(0);
    assert_eq!(rng.permutation(10), vec![2, 8, 4, 9, 1, 6, 7, 3, 0, 5]);
}

#[test]
fn shuffle_matches_numpy() {
    let mut rng = NumpyRandomState::from_seed(42);
    let mut a: Vec<i64> = (0..12).collect();
    rng.shuffle(&mut a);
    assert_eq!(a, vec![10, 9, 0, 8, 5, 2, 1, 11, 4, 7, 3, 6]);
}

#[test]
fn randint_matches_numpy() {
    // The masked-rejection loop `randint` shares with `shuffle`: a divergence
    // here (e.g. a modulo reduction) still produces uniform values, so only an
    // exact comparison catches it.
    let mut rng = NumpyRandomState::from_seed(42);
    let got: Vec<u64> = (0..6).map(|_| rng.randint(7)).collect();
    assert_eq!(got, vec![6, 3, 4, 6, 2, 4]);
}

#[test]
fn state_round_trip_resumes_the_same_stream() {
    // The PyO3 layer hands a caller's `RandomState` words in and writes the
    // advanced words back; this asserts that round trip is lossless.
    let mut rng = NumpyRandomState::from_seed(7);
    let _ = rng.permutation(5);
    let saved = NumpyRandomState::from_key(*rng.key(), rng.pos());

    let mut a = rng;
    let mut b = saved;
    assert_eq!(a.permutation(9), b.permutation(9));
}

#[test]
fn sample_without_replacement_auto_uses_permutation_for_mid_ratios() {
    // ratio 5/10 = 0.5 is inside (0.01, 0.99), so sklearn short-circuits to
    // `permutation(n)[:k]` — the same draw as a bare permutation.
    let mut rng = NumpyRandomState::from_seed(42);
    let got = sample_without_replacement(10, 5, SampleMethod::Auto, &mut rng).expect("valid");
    let mut check = NumpyRandomState::from_seed(42);
    assert_eq!(got, check.permutation(10)[..5].to_vec());
}

#[test]
fn sample_without_replacement_rejects_oversized_requests() {
    let mut rng = NumpyRandomState::from_seed(0);
    assert!(sample_without_replacement(3, 4, SampleMethod::Auto, &mut rng).is_none());
}

#[test]
fn sample_without_replacement_methods_are_distinct_but_valid() {
    // Each method walks the stream differently, so they must NOT agree; what
    // they share is that every result is a valid sample.
    let mut seen = Vec::new();
    for method in [
        SampleMethod::TrackingSelection,
        SampleMethod::ReservoirSampling,
        SampleMethod::Pool,
    ] {
        let mut rng = NumpyRandomState::from_seed(3);
        let got = sample_without_replacement(50, 6, method, &mut rng).expect("valid");
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|&v| (0..50).contains(&v)));
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 6, "{method:?} produced a duplicate");
        seen.push(got);
    }
    assert!(
        seen[0] != seen[1] || seen[1] != seen[2],
        "the three methods collapsed to one stream"
    );
}

#[test]
fn approximate_mode_matches_sklearn_doctests() {
    // The four examples in `sklearn.utils.extmath._approximate_mode`'s
    // docstring, including the two that differ only by seed — those are the
    // ones that pin the random tie-break rather than the flooring.
    let mut rng = NumpyRandomState::from_seed(0);
    assert_eq!(approximate_mode(&[4, 2], 3, &mut rng), vec![2, 1]);

    let mut rng = NumpyRandomState::from_seed(0);
    assert_eq!(approximate_mode(&[5, 2], 4, &mut rng), vec![3, 1]);

    let mut rng = NumpyRandomState::from_seed(0);
    assert_eq!(
        approximate_mode(&[2, 2, 2, 1], 2, &mut rng),
        vec![0, 1, 1, 0]
    );

    let mut rng = NumpyRandomState::from_seed(42);
    assert_eq!(
        approximate_mode(&[2, 2, 2, 1], 2, &mut rng),
        vec![1, 1, 0, 0]
    );
}
