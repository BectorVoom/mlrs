//! TSNE-PARAMS — oracle gates for every STRING-valued `TSNE` parameter, plus
//! the value-neutrality gates for the parameters that only move the wall clock.
//!
//! ## The oracle strategy, per string parameter
//!
//! **`metric` (22 values) gets a real sklearn VALUE oracle, twice over.** The
//! input-space distance matrix is a deterministic function of the design, so
//! [`metrics_match_sklearn`] asserts mlrs's squared distances against
//! `sklearn.metrics.pairwise_distances` at the ≤1e-5 contract for every metric
//! string. That pins the formula. But a port can get the formula right and
//! still consume it wrongly, so [`joint_p_matches_sklearn_per_metric`] asserts
//! the DENSE joint-probability matrix too — which additionally pins the
//! `distances **= 2` that sklearn applies to every metric except
//! `'euclidean'`, and the `float32` rounding inside the perplexity search.
//! Neither gate is a band; both are exact-to-tolerance.
//!
//! **`method` and `init` are gated from a SHARED INJECTED INIT.** t-SNE's
//! descent is a thousand chaotic iterations, so two runs that start from
//! different embeddings cannot be compared at the value level, and a band gate
//! over a stochastic init proves little. The fixture therefore records
//! sklearn's result for each `method` and each `init` starting from ONE
//! recorded `init_array`; [`method_reaches_sklearn_band`] and
//! [`init_array_reaches_sklearn_band`] hold mlrs to sklearn's own
//! neighbourhood preservation and KL from that same starting point, which is
//! the tightest comparison the dynamics permit.
//!
//! **`learning_rate='auto'` is gated by EXACT equality.** It is not a separate
//! algorithm, it is the single number `max(n / early_exaggeration / 4, 50)`.
//! [`learning_rate_auto_resolves_to_the_sklearn_formula`] fits twice — once
//! with `'auto'`, once with that number spelled out — and requires
//! bit-identical embeddings. A band would not distinguish `'auto'` from any
//! nearby constant; equality does.
//!
//! **`n_jobs` and `verbose` are gated by EXACT equality too**, for the opposite
//! reason: they are provably value-neutral here (every parallel reduction runs
//! in point order), so anything less than bit-identical output is a bug.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::manifold::tsne::{
    joint_probabilities, LearningRate, Tsne, TsneInit, TsneMethod,
};
use mlrs_algos::manifold::tsne_metric::{
    pairwise_squared, resolve_metric_params, MetricParams, TsneMetric,
};
use mlrs_algos::typestate::Fit;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry (`gen_oracle.py::gen_tsne_metrics` / `gen_tsne_params`).
const N: usize = 48;
const P: usize = 5;
const D: usize = 2;
const TRUST_K: usize = 5;
/// The ≤1e-5 milestone contract, applied to the deterministic tiers.
const TOL_ABS: f64 = 1e-5;
const TOL_REL: f64 = 1e-5;

/// Every metric the fixture carries a `D_*` / `P_*` pair for, on the shared
/// 5-feature design. `haversine`, `nan_euclidean` and `precomputed` have their
/// own designs and are gated separately.
const GENERIC_METRICS: &[&str] = &[
    "euclidean",
    "l2",
    "sqeuclidean",
    "l1",
    "manhattan",
    "cityblock",
    "chebyshev",
    "minkowski",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "seuclidean",
    "mahalanobis",
    "hamming",
    "matching",
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
];

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn metrics_case() -> OracleCase {
    load_npz(fixture("tsne_metrics_f64_seed42.npz")).expect("tsne_metrics fixture loads")
}

fn params_case() -> OracleCase {
    load_npz(fixture("tsne_params_f64_seed42.npz")).expect("tsne_params fixture loads")
}

fn assert_close(got: &[f64], expected: &[f64], what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        // A NaN in the reference must be a NaN in the port (nan_euclidean with
        // no shared coordinate, dice on two all-zero rows) — the degeneracies
        // are mirrored, not repaired, so they are asserted rather than skipped.
        if e.is_nan() {
            assert!(g.is_nan(), "{what}: expected NaN at {i}, got {g}");
            continue;
        }
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= TOL_ABS + TOL_REL * e.abs(),
            "{what}: mismatch at {i}: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// sklearn's `trustworthiness(X, emb, n_neighbors=k)`, ported so the gate does
/// not depend on a Python round trip.
fn trustworthiness(x: &[f64], emb: &[f64], n: usize, p: usize, d: usize, k: usize) -> f64 {
    let sq = |a: &[f64], i: usize, j: usize, w: usize| -> f64 {
        (0..w)
            .map(|t| {
                let v = a[i * w + t] - a[j * w + t];
                v * v
            })
            .sum()
    };
    // Rank of j among i's neighbours in the INPUT space (1-based, self excluded).
    let mut rank = vec![0usize; n * n];
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            sq(x, i, a, p)
                .partial_cmp(&sq(x, i, b, p))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for (r, &j) in order.iter().enumerate() {
            rank[i * n + j] = r + 1;
        }
    }
    let mut t = 0.0f64;
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            sq(emb, i, a, d)
                .partial_cmp(&sq(emb, i, b, d))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &j in order.iter().take(k) {
            let r = rank[i * n + j] as f64 - k as f64;
            if r > 0.0 {
                t += r;
            }
        }
    }
    let nf = n as f64;
    let kf = k as f64;
    1.0 - t * (2.0 / (nf * kf * (2.0 * nf - 3.0 * kf - 1.0)))
}

/// `Tsne` carries a `DeviceArray` and so has no `Debug` impl (the family
/// precedent), which rules out `Result::expect_err`. These unwrap the error
/// side by hand.
fn expect_build_err(b: mlrs_algos::manifold::tsne::TsneBuilder) -> BuildError {
    match b.build::<f64>() {
        Ok(_) => panic!("expected the builder to reject these hyperparameters"),
        Err(e) => e,
    }
}

fn expect_fit_err(
    est: Tsne<f64, mlrs_algos::typestate::Unfit>,
    x: &[f64],
    n: usize,
    p: usize,
) -> AlgoError {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, x);
    match est.fit(&mut pool, &xd, None, (n, p)) {
        Ok(_) => panic!("expected fit to reject this configuration"),
        Err(e) => e,
    }
}

/// Fit `est` on the host design `x` and return `(embedding, kl, n_iter)`.
fn run_fit(est: Tsne<f64, mlrs_algos::typestate::Unfit>, x: &[f64], n: usize, p: usize) -> (Vec<f64>, f64, usize) {
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, x);
    let fitted = est.fit(&mut pool, &xd, None, (n, p)).expect("fit succeeds");
    let emb = fitted.embedding(&pool);
    (emb, fitted.kl_divergence(), fitted.n_iter())
}

// ===========================================================================
// `metric` — deterministic VALUE gates
// ===========================================================================

/// Tier 1 for `metric`: the squared input distances match sklearn's
/// `pairwise_distances` for every string, at ≤1e-5.
///
/// The expected value is `D²` for every metric EXCEPT `'euclidean'`, whose
/// fixture entry sklearn already emitted squared (`squared=True`) — that
/// asymmetry is sklearn's own, and reproducing it is half of what this gate
/// checks.
#[test]
fn metrics_match_sklearn() {
    let case = metrics_case();
    let x = case.f64("X").expect("X").to_vec();

    for &name in GENERIC_METRICS {
        let metric = TsneMetric::from_sklearn_name(name)
            .unwrap_or_else(|| panic!("{name} must parse"));
        let d_ref = case
            .f64(&format!("D_{name}"))
            .unwrap_or_else(|| panic!("fixture must carry D_{name}"));
        let expected: Vec<f64> = if name == "euclidean" {
            d_ref.to_vec()
        } else {
            d_ref.iter().map(|v| v * v).collect()
        };
        let rp = resolve_metric_params(&x, N, P, metric, &MetricParams::default())
            .expect("metric params resolve");
        let got = pairwise_squared(&x, N, P, metric, &rp, 4).expect("pairwise succeeds");
        assert_close(&got, &expected, &format!("metric='{name}' squared distances"));
    }
}

/// The three metrics with their own design geometry: `haversine` (exactly 2
/// features), `nan_euclidean` (missing entries), and `precomputed` (X IS the
/// distance matrix — built from cityblock in the fixture, so an implementation
/// that silently recomputes euclidean distances fails here).
#[test]
fn special_geometry_metrics_match_sklearn() {
    let case = metrics_case();

    for (key, name, cols) in [
        ("Xh", "haversine", 2usize),
        ("Xnan", "nan_euclidean", P),
        ("Xpre", "precomputed", N),
    ] {
        let x = case.f64(key).unwrap_or_else(|| panic!("fixture must carry {key}"));
        let metric = TsneMetric::from_sklearn_name(name).expect("parses");
        let d_ref = case
            .f64(&format!("D_{name}"))
            .unwrap_or_else(|| panic!("fixture must carry D_{name}"));
        let expected: Vec<f64> = d_ref.iter().map(|v| v * v).collect();
        let rp = resolve_metric_params(x, N, cols, metric, &MetricParams::default())
            .expect("metric params resolve");
        let got = pairwise_squared(x, N, cols, metric, &rp, 4).expect("pairwise succeeds");
        assert_close(&got, &expected, &format!("metric='{name}' squared distances"));
    }
}

/// Tier 2 for `metric`: the DENSE joint-probability matrix `P` matches
/// sklearn's, per metric. This is what proves the distances are consumed the
/// way sklearn consumes them (the f32 rounding, the 100-step bisection, the
/// symmetrize-and-normalize), not merely computed the same way.
#[test]
fn joint_p_matches_sklearn_per_metric() {
    let case = metrics_case();
    let x = case.f64("X").expect("X").to_vec();
    let perplexity = case.f64("perplexity").expect("perplexity")[0];

    for &name in GENERIC_METRICS {
        let metric = TsneMetric::from_sklearn_name(name).expect("parses");
        let p_ref = case
            .f64(&format!("P_{name}"))
            .unwrap_or_else(|| panic!("fixture must carry P_{name}"));
        let rp = resolve_metric_params(&x, N, P, metric, &MetricParams::default())
            .expect("metric params resolve");
        let dsq = pairwise_squared(&x, N, P, metric, &rp, 4).expect("pairwise succeeds");
        let got = joint_probabilities(&dsq, N, perplexity, 4);
        assert_close(&got, p_ref, &format!("metric='{name}' joint P"));
    }
}

/// `l2`/`l1`/`cityblock`/`matching` are ALIASES the parser collapses. Collapsing
/// is only sound if sklearn agrees they are the same metric — assert that on
/// the fixture rather than trusting the docs.
#[test]
fn metric_aliases_agree_with_their_canonical_name() {
    let case = metrics_case();
    for (alias, canonical) in [
        ("l2", "euclidean"),
        ("l1", "manhattan"),
        ("cityblock", "manhattan"),
        ("matching", "hamming"),
    ] {
        assert_eq!(
            TsneMetric::from_sklearn_name(alias),
            TsneMetric::from_sklearn_name(canonical),
            "{alias} must parse to the same variant as {canonical}"
        );
        let a = case.f64(&format!("D_{alias}")).expect("alias D");
        let c = case.f64(&format!("D_{canonical}")).expect("canonical D");
        // `l2` takes sklearn's sqrt-then-square round trip where `euclidean`
        // asks for the squared value directly, so compare in the SQUARED
        // domain the estimator actually consumes.
        let (a2, c2): (Vec<f64>, Vec<f64>) = if alias == "l2" {
            (a.iter().map(|v| v * v).collect(), c.to_vec())
        } else {
            (a.to_vec(), c.to_vec())
        };
        assert_close(&a2, &c2, &format!("sklearn '{alias}' vs '{canonical}'"));
    }
}

/// Every string in sklearn's `StrOptions` set parses, and nothing else does.
#[test]
fn metric_string_surface_matches_sklearn() {
    for &name in GENERIC_METRICS {
        assert!(
            TsneMetric::from_sklearn_name(name).is_some(),
            "sklearn accepts metric='{name}', so mlrs must parse it"
        );
    }
    for name in ["haversine", "nan_euclidean", "precomputed", "wminkowski"] {
        assert!(
            TsneMetric::from_sklearn_name(name).is_some(),
            "sklearn accepts metric='{name}'"
        );
    }
    for name in ["", "Euclidean", "l3", "sokalmichener", "not_a_metric"] {
        assert!(
            TsneMetric::from_sklearn_name(name).is_none(),
            "metric='{name}' is outside sklearn's StrOptions and must be rejected"
        );
    }
}

/// `wminkowski` is in sklearn's `StrOptions` but scipy REMOVED the metric, so
/// sklearn accepts it at construction and then fails at fit. mlrs mirrors the
/// shape of that: the string parses, and evaluating it is a typed error.
#[test]
fn wminkowski_parses_then_fails_at_fit() {
    let metric = TsneMetric::from_sklearn_name("wminkowski").expect("parses like sklearn's");
    let x = vec![0.0f64; N * P];
    let rp = resolve_metric_params(&x, N, P, metric, &MetricParams::default()).expect("resolve");
    let err = pairwise_squared(&x, N, P, metric, &rp, 1).expect_err("must fail like sklearn");
    assert!(
        matches!(err, AlgoError::InvalidGraphInput { .. }),
        "wminkowski must fail with a typed error, got {err:?}"
    );
}

/// `haversine` is only defined on 2 features and `precomputed` needs a square
/// `X`. Both are rejected BEFORE any `O(n²)` work.
#[test]
fn metric_geometry_is_validated() {
    let x = vec![0.0f64; N * P];
    let rp = resolve_metric_params(&x, N, P, TsneMetric::Haversine, &MetricParams::default())
        .expect("resolve");
    assert!(
        pairwise_squared(&x, N, P, TsneMetric::Haversine, &rp, 1).is_err(),
        "haversine on {P} features must be rejected"
    );
    assert!(
        pairwise_squared(&x, N, P, TsneMetric::Precomputed, &rp, 1).is_err(),
        "a non-square precomputed X must be rejected"
    );
}

/// `metric_params` carries scipy's keywords. `minkowski(p=2)` must reproduce
/// euclidean exactly, and `p=1` manhattan — the cheapest possible proof that
/// the keyword reaches the pair loop rather than being stored and ignored.
#[test]
fn metric_params_p_reaches_the_pair_loop() {
    let case = metrics_case();
    let x = case.f64("X").expect("X").to_vec();
    for (p_value, equivalent) in [(2.0, "euclidean"), (1.0, "manhattan")] {
        let params = MetricParams {
            p: Some(p_value),
            ..MetricParams::default()
        };
        let rp = resolve_metric_params(&x, N, P, TsneMetric::Minkowski, &params).expect("resolve");
        let got = pairwise_squared(&x, N, P, TsneMetric::Minkowski, &rp, 4).expect("pairwise");
        let d_ref = case.f64(&format!("D_{equivalent}")).expect("reference D");
        let expected: Vec<f64> = if equivalent == "euclidean" {
            d_ref.to_vec()
        } else {
            d_ref.iter().map(|v| v * v).collect()
        };
        assert_close(&got, &expected, &format!("minkowski(p={p_value})"));
    }
}

// ===========================================================================
// `method` — band gate from a shared injected init
// ===========================================================================

/// Both `method` values reach sklearn's own neighbourhood preservation and KL
/// from the SAME injected starting embedding.
#[test]
fn method_reaches_sklearn_band() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let init = case.f64("init_array").expect("init_array").to_vec();
    let perplexity = case.f64("perplexity").expect("perplexity")[0];

    for (method, tag) in [
        (TsneMethod::BarnesHut, "barnes_hut"),
        (TsneMethod::Exact, "exact"),
    ] {
        let est = Tsne::<f64>::builder()
            .perplexity(perplexity)
            .method(method)
            .init(TsneInit::Array(init.clone()))
            .seed(42)
            .build::<f64>()
            .expect("valid hyperparameters");
        let (emb, kl, _n_iter) = run_fit(est, &x, N, P);

        let trust = trustworthiness(&x, &emb, N, P, D, TRUST_K);
        let trust_ref = case.f64(&format!("trust_{tag}")).expect("trust ref")[0];
        let kl_ref = case.f64(&format!("kl_{tag}")).expect("kl ref")[0];
        assert!(
            trust >= trust_ref - 0.05,
            "method='{tag}': trustworthiness {trust} below sklearn's {trust_ref} - 0.05"
        );
        assert!(
            kl > 0.0 && kl <= kl_ref + 0.25,
            "method='{tag}': kl {kl} outside (0, {kl_ref} + 0.25]"
        );
    }
}

/// `method='barnes_hut'` needs a quad-/oct-tree, so sklearn caps
/// `n_components <= 3`. Rejected at `build()` — the pair is knowable without
/// data (the D-08 split).
#[test]
fn barnes_hut_rejects_more_than_three_components() {
    let err = expect_build_err(
        Tsne::<f64>::builder()
            .method(TsneMethod::BarnesHut)
            .n_components(4),
    );
    assert!(
        matches!(err, BuildError::InvalidNComponents { .. }),
        "expected InvalidNComponents, got {err:?}"
    );
    // The exact method has no such cap.
    Tsne::<f64>::builder()
        .method(TsneMethod::Exact)
        .n_components(4)
        .build::<f64>()
        .expect("exact with n_components=4 is legal");
}

// ===========================================================================
// `init` — band gate + the deterministic array form
// ===========================================================================

/// All three `init` forms reach sklearn's band. The array form is the tightest
/// of the three: mlrs and sklearn start from the identical embedding, so any
/// divergence is arithmetic rather than initialization.
#[test]
fn init_array_reaches_sklearn_band() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let init_arr = case.f64("init_array").expect("init_array").to_vec();
    let perplexity = case.f64("perplexity").expect("perplexity")[0];

    for (init, tag) in [
        (TsneInit::Array(init_arr.clone()), "init_array"),
        (TsneInit::Pca, "init_pca"),
        (TsneInit::Random, "init_random"),
    ] {
        let est = Tsne::<f64>::builder()
            .perplexity(perplexity)
            .method(TsneMethod::Exact)
            .init(init)
            .seed(42)
            .build::<f64>()
            .expect("valid hyperparameters");
        let (emb, kl, _) = run_fit(est, &x, N, P);
        let trust = trustworthiness(&x, &emb, N, P, D, TRUST_K);
        let trust_ref = case.f64(&format!("trust_{tag}")).expect("trust ref")[0];
        let kl_ref = case.f64(&format!("kl_{tag}")).expect("kl ref")[0];
        assert!(
            trust >= trust_ref - 0.05,
            "init={tag}: trustworthiness {trust} below sklearn's {trust_ref} - 0.05"
        );
        assert!(
            kl > 0.0 && kl <= kl_ref + 0.25,
            "init={tag}: kl {kl} outside (0, {kl_ref} + 0.25]"
        );
    }
}

/// An `init` array whose length does not match `(n_samples, n_components)` is a
/// DATA-dependent error, so it is rejected at `fit` rather than `build`.
#[test]
fn init_array_shape_is_validated() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let est = Tsne::<f64>::builder()
        .perplexity(10.0)
        .method(TsneMethod::Exact)
        .init(TsneInit::Array(vec![0.0; N * D - 1]))
        .build::<f64>()
        .expect("build does not see the data");
    let err = expect_fit_err(est, &x, N, P);
    assert!(
        matches!(err, AlgoError::InvalidGraphInput { .. }),
        "expected InvalidGraphInput, got {err:?}"
    );
}

/// sklearn: `'The parameter init="pca" cannot be used with
/// metric="precomputed".'` There is no feature space to project.
#[test]
fn pca_init_is_rejected_with_precomputed() {
    let case = metrics_case();
    let xpre = case.f64("Xpre").expect("Xpre").to_vec();
    let est = Tsne::<f64>::builder()
        .perplexity(10.0)
        .method(TsneMethod::Exact)
        .metric(TsneMetric::Precomputed)
        .init(TsneInit::Pca)
        .build::<f64>()
        .expect("build does not see the data");
    let err = expect_fit_err(est, &xpre, N, N);
    assert!(
        matches!(err, AlgoError::InvalidGraphInput { .. }),
        "expected InvalidGraphInput, got {err:?}"
    );
}

// ===========================================================================
// `learning_rate='auto'` — EXACT equality against the resolved constant
// ===========================================================================

/// `'auto'` is `max(n_samples / early_exaggeration / 4, 50)` and nothing else.
/// Fitting with the sentinel and with that number spelled out must produce
/// BIT-IDENTICAL embeddings — a band would not tell `'auto'` apart from any
/// nearby constant.
#[test]
fn learning_rate_auto_resolves_to_the_sklearn_formula() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let lr_auto = case.f64("lr_auto").expect("lr_auto")[0];
    // The fixture's value is sklearn's own; re-derive it here so the test
    // fails if either side drifts.
    let derived = (N as f64 / 12.0 / 4.0).max(50.0);
    assert_eq!(
        lr_auto, derived,
        "the fixture's resolved auto learning rate must equal sklearn's formula"
    );

    let build = |lr: LearningRate| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .method(TsneMethod::Exact)
            .init(TsneInit::Pca)
            .learning_rate(lr)
            .max_iter(300)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (auto, kl_auto, it_auto) = run_fit(build(LearningRate::Auto), &x, N, P);
    let (explicit, kl_explicit, it_explicit) =
        run_fit(build(LearningRate::Value(derived)), &x, N, P);

    assert_eq!(
        auto, explicit,
        "learning_rate='auto' must be bit-identical to the resolved constant"
    );
    assert_eq!(kl_auto, kl_explicit, "kl_divergence_ must match exactly");
    assert_eq!(it_auto, it_explicit, "n_iter_ must match exactly");
}

// ===========================================================================
// Value-NEUTRAL parameters — gated by exact equality
// ===========================================================================

/// `n_jobs` only picks a worker count. Every parallel pass here reduces in
/// POINT order, so the count cannot reach a value — assert that directly,
/// across a spread that covers the serial arm, an even split, and joblib's
/// negative offsets.
#[test]
fn n_jobs_is_value_neutral() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let build = |n_jobs: Option<i32>| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .init(TsneInit::Pca)
            .max_iter(300)
            .n_jobs(n_jobs)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (base, base_kl, _) = run_fit(build(Some(1)), &x, N, P);
    for n_jobs in [Some(2), Some(4), Some(-1), Some(-2), None] {
        let (emb, kl, _) = run_fit(build(n_jobs), &x, N, P);
        assert_eq!(
            emb, base,
            "n_jobs={n_jobs:?} changed the embedding; the reductions are not order-stable"
        );
        assert_eq!(kl, base_kl, "n_jobs={n_jobs:?} changed kl_divergence_");
    }
}

/// `n_jobs = 0` names no worker count; joblib itself rejects it.
#[test]
fn n_jobs_zero_is_rejected() {
    let err = expect_build_err(Tsne::<f64>::builder().n_jobs(Some(0)));
    assert!(
        matches!(err, BuildError::InvalidNJobs { .. }),
        "expected InvalidNJobs, got {err:?}"
    );
}

/// `verbose` only prints. It must not perturb the fit.
#[test]
fn verbose_is_value_neutral() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let build = |verbose: usize| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .init(TsneInit::Pca)
            .max_iter(300)
            .verbose(verbose)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (quiet, quiet_kl, _) = run_fit(build(0), &x, N, P);
    for v in [1, 2, 5] {
        let (loud, loud_kl, _) = run_fit(build(v), &x, N, P);
        assert_eq!(loud, quiet, "verbose={v} changed the embedding");
        assert_eq!(loud_kl, quiet_kl, "verbose={v} changed kl_divergence_");
    }
}

// ===========================================================================
// The remaining numeric parameters
// ===========================================================================

/// `angle` is Barnes-Hut's accuracy/speed dial. At `angle = 0` no cell can ever
/// satisfy `width² / dist² < 0`, so every leaf is visited individually and the
/// negative force becomes the EXACT `O(n²)` summation. That gives the parameter
/// a checkable endpoint: `barnes_hut(angle=0)` must land in the same KL
/// neighbourhood as `exact` on the same init, which no band over intermediate
/// angles could establish.
#[test]
fn angle_zero_degrades_barnes_hut_towards_exact() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let init = case.f64("init_array").expect("init_array").to_vec();

    let build = |method: TsneMethod, angle: f64| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .method(method)
            .angle(angle)
            .init(TsneInit::Array(init.clone()))
            .max_iter(300)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (_, kl_exact, _) = run_fit(build(TsneMethod::Exact, 0.5), &x, N, P);
    let (_, kl_bh0, _) = run_fit(build(TsneMethod::BarnesHut, 0.0), &x, N, P);
    let (_, kl_bh_coarse, _) = run_fit(build(TsneMethod::BarnesHut, 1.0), &x, N, P);

    // `barnes_hut` still uses a SPARSE k-NN P where `exact` uses the dense one,
    // so the two KLs are not the same number even at angle 0 — but the
    // exact-summation arm must be the closer of the two.
    let near = (kl_bh0 - kl_exact).abs();
    let far = (kl_bh_coarse - kl_exact).abs();
    assert!(
        near <= far,
        "angle=0 (exact summation) should track the exact method at least as \
         closely as angle=1: |{kl_bh0} - {kl_exact}| = {near} vs {far}"
    );
}

/// sklearn's `angle` constraint is `Interval(Real, 0, 1, closed='both')`.
#[test]
fn angle_range_is_validated() {
    for good in [0.0, 0.5, 1.0] {
        Tsne::<f64>::builder()
            .angle(good)
            .build::<f64>()
            .unwrap_or_else(|e| panic!("angle={good} must be accepted, got {e:?}"));
    }
    for bad in [-0.1, 1.0001, f64::NAN] {
        let err = expect_build_err(Tsne::<f64>::builder().angle(bad));
        assert!(
            matches!(err, BuildError::InvalidAngle { .. }),
            "expected InvalidAngle for angle={bad}, got {err:?}"
        );
    }
}

/// `n_iter_without_progress` is the MAIN phase's patience. Squeezing it to 1
/// must stop the descent no later than the default does — the parameter has to
/// reach the loop, not merely be stored.
#[test]
fn n_iter_without_progress_bounds_the_descent() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let build = |patience: usize| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .method(TsneMethod::Exact)
            .init(TsneInit::Pca)
            .n_iter_without_progress(patience)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (_, _, it_tight) = run_fit(build(1), &x, N, P);
    let (_, _, it_default) = run_fit(build(300), &x, N, P);
    assert!(
        it_tight <= it_default,
        "n_iter_without_progress=1 ran {it_tight} iterations, more than the \
         default's {it_default}"
    );
}

/// `min_grad_norm` is the other stopping rule, and its stop count is EXACTLY
/// checkable rather than merely bounded.
///
/// A threshold no gradient can exceed trips at the first convergence check of
/// EACH of the two phases — checks run at `(i + 1) % 50 == 0`, so the
/// exploration phase breaks at `i = 49`, phase 2 resumes at `i = 50` and breaks
/// at `i = 99`. `n_iter_ = 99` is therefore a structural consequence of the
/// two-phase schedule, and it is the number sklearn 1.9.0 reports for this
/// fixture (verified against the installed version, not assumed). A `< 60`
/// bound would have looked like a failure here and hidden the real invariant.
#[test]
fn min_grad_norm_stops_the_descent_at_sklearns_iteration() {
    let case = params_case();
    let x = case.f64("X").expect("X").to_vec();
    let build = |min_grad_norm: f64| {
        Tsne::<f64>::builder()
            .perplexity(10.0)
            .method(TsneMethod::Exact)
            .init(TsneInit::Pca)
            .min_grad_norm(min_grad_norm)
            .build::<f64>()
            .expect("valid hyperparameters")
    };
    let (_, _, stopped) = run_fit(build(1e9), &x, N, P);
    assert_eq!(
        stopped, 99,
        "min_grad_norm=1e9 must break at the first check of each phase, as sklearn does"
    );

    // And the default threshold must NOT stop early, or the gate above would
    // pass for a fit that simply never runs.
    let (_, _, full) = run_fit(build(1e-7), &x, N, P);
    assert_eq!(
        full, 999,
        "the default min_grad_norm must let the descent run its full budget"
    );
}
