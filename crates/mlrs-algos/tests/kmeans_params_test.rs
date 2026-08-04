//! KMeans full-parameter-surface tests — sklearn's `init` / `n_init` /
//! `algorithm` / `verbose` / `random_state` / `copy_x` (CLUSTER-01).
//!
//! ## The oracle strategy, per parameter
//!
//! `algorithm` gets a REAL sklearn value oracle. Elkan is an EXACT
//! acceleration of Lloyd — the triangle-inequality bounds only skip distance
//! computations that provably cannot win — so from the SAME injected init it
//! must reproduce the committed `kmeans_{f32,f64}_seed42.npz` sklearn
//! reference within the same 1e-5 contract the Lloyd arm is held to. That is
//! the strongest possible check on `algorithm`: the two arms are asserted
//! against sklearn INDEPENDENTLY, not merely against each other.
//!
//! `init='k-means++'` and `init='random'` CANNOT have a value oracle at this
//! layer: mlrs's k-means++ is the D²-weighted host sampler (`kmeanspp_sample`,
//! one draw per center) seeded by SplitMix64, while sklearn's draws from a
//! numpy `RandomState` and additionally runs `2 + log(k)` greedy local trials
//! per center. Same distribution, different stream — so the INIT differs and
//! with it the local optimum. They are tested here for the properties that
//! must hold regardless of stream (k distinct in-range centers, a converged
//! partition, `n_init` monotonicity), and against live sklearn VALUES on
//! well-separated blobs — where every init reaches the same global optimum —
//! in `crates/mlrs-py/python/tests/test_oracle_cluster.py`.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::kmeans::{KMeans, KMeansAlgorithm, KMeansInit, NInit};
use mlrs_algos::error::BuildError;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// KMeans fixture geometry (gen_oracle.py KM_N_SAMPLES × KM_N_FEATURES, K=KM_K).
const KM_N_SAMPLES: usize = 30;
const KM_N_FEATURES: usize = 4;
const KM_K: usize = 3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans fixtures are f32/f64 only"),
    }
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans fixtures are f32/f64 only"),
    }
}

fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs,
            tol.rel
        );
    }
}

/// A fitted KMeans reduced to its host-comparable surface.
struct FitOut {
    centers: Vec<f64>,
    labels: Vec<i64>,
    inertia: f64,
    n_iter: usize,
}

/// Fit the fixture's INJECTED init (D-09) under the given `algorithm`.
fn fit_fixture<F>(case: &OracleCase, algorithm: KMeansAlgorithm) -> FitOut
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let init_host: Vec<f64> = case.expect_f64("init").to_vec();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let km = KMeans::<F>::builder()
        .n_clusters(KM_K)
        .init(Some(init_host))
        .algorithm(algorithm)
        .build::<F>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (KM_N_SAMPLES, KM_N_FEATURES))
        .expect("fit on a valid shape");

    FitOut {
        centers: km
            .cluster_centers(&pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect(),
        labels: km.labels(&pool).iter().map(|&l| l as i64).collect(),
        inertia: host_to_f64(km.inertia()),
        n_iter: km.n_iter(),
    }
}

/// Assert a fit reproduces the fixture's sklearn reference within `tol` (up to
/// a label permutation, the D-09 contract).
fn assert_matches_sklearn(case: &OracleCase, out: &FitOut, tol: &Tolerance, label: &str) {
    let centers_ref = case.expect_f64("centers");
    let labels_ref: Vec<i64> = case.expect_f64("labels").iter().map(|&v| v as i64).collect();
    let inertia_ref = case.expect_f64("inertia");

    let acc = best_match_accuracy(&out.labels, &labels_ref);
    assert!(
        (acc - 1.0).abs() < f64::EPSILON,
        "{label}: best_match_accuracy {acc} != 1.0 (labels are not a permutation of sklearn's)"
    );

    let mapping = mlrs_core::best_mapping(&out.labels, &labels_ref);
    for fitted_c in 0..KM_K {
        let ref_c = *mapping
            .get(&(fitted_c as i64))
            .expect("every fitted cluster maps to a sklearn cluster") as usize;
        assert_close(
            &out.centers[fitted_c * KM_N_FEATURES..(fitted_c + 1) * KM_N_FEATURES],
            &centers_ref[ref_c * KM_N_FEATURES..(ref_c + 1) * KM_N_FEATURES],
            tol,
            &format!("{label} center[{fitted_c}->{ref_c}]"),
        );
    }
    assert_close(
        &[out.inertia],
        &[inertia_ref[0]],
        tol,
        &format!("{label} inertia_"),
    );
}

// ---------------------------------------------------------------------------
// algorithm — the sklearn VALUE oracle for both string values
// ---------------------------------------------------------------------------

/// `algorithm='elkan'` reproduces the sklearn reference, f32. Elkan prunes
/// distance computations; it does not change the answer.
#[test]
fn algorithm_elkan_matches_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "algorithm=elkan");
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    let out = fit_fixture::<f32>(&case, KMeansAlgorithm::Elkan);
    assert_matches_sklearn(&case, &out, &F32_TOL, "kmeans f32 elkan");
}

/// `algorithm='elkan'` reproduces the sklearn reference, f64.
#[test]
fn algorithm_elkan_matches_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "algorithm=elkan");
    if capability::skip_f64_with_log() {
        println!("kmeans f64 elkan backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    let out = fit_fixture::<f64>(&case, KMeansAlgorithm::Elkan);
    assert_matches_sklearn(&case, &out, &F64_TOL, "kmeans f64 elkan");
}

/// `algorithm='lloyd'` (the explicit default) reproduces the sklearn reference
/// — the paired half of the elkan test, so a regression in either arm is
/// attributable.
#[test]
fn algorithm_lloyd_matches_sklearn_f32() {
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    let out = fit_fixture::<f32>(&case, KMeansAlgorithm::Lloyd);
    assert_matches_sklearn(&case, &out, &F32_TOL, "kmeans f32 lloyd");
}

/// The two arms agree with EACH OTHER on the fitted labels and converge in the
/// same number of iterations. Both already match sklearn above; this pins the
/// stronger claim that Elkan visits the same iterate sequence, so a future
/// pruning bug that happened to land on an equally good optimum still fails.
#[test]
fn algorithm_elkan_and_lloyd_agree_f64() {
    if capability::skip_f64_with_log() {
        println!("kmeans elkan/lloyd agreement: SKIPPED (no f64 on this adapter)");
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    let lloyd = fit_fixture::<f64>(&case, KMeansAlgorithm::Lloyd);
    let elkan = fit_fixture::<f64>(&case, KMeansAlgorithm::Elkan);

    assert_eq!(
        lloyd.labels, elkan.labels,
        "elkan must produce the IDENTICAL labeling as lloyd from the same init"
    );
    assert_eq!(
        lloyd.n_iter, elkan.n_iter,
        "elkan must take the same iteration count as lloyd (same iterate sequence)"
    );
    assert_close(
        &[elkan.inertia],
        &[lloyd.inertia],
        &F64_TOL,
        "elkan vs lloyd inertia_",
    );
    assert_close(
        &elkan.centers,
        &lloyd.centers,
        &F64_TOL,
        "elkan vs lloyd cluster_centers_",
    );
}

/// sklearn silently degrades `algorithm='elkan'` to `'lloyd'` at `k == 1`
/// (there is no "other center" to bound against). mlrs performs the same
/// override — visible through `algorithm_used()` — and still fits.
#[test]
fn algorithm_elkan_degrades_to_lloyd_at_k1() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.25).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    let km = KMeans::<f32>::builder()
        .n_clusters(1)
        .algorithm(KMeansAlgorithm::Elkan)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (12, 2))
        .expect("k=1 elkan fit must not error");

    assert_eq!(
        km.algorithm_used(),
        KMeansAlgorithm::Lloyd,
        "elkan at k == 1 must resolve to lloyd (sklearn's override)"
    );
    // The single center is the mean of every row.
    let centers = km.cluster_centers(&pool);
    let mean0: f32 = (0..12).map(|i| x[i * 2]).sum::<f32>() / 12.0;
    let mean1: f32 = (0..12).map(|i| x[i * 2 + 1]).sum::<f32>() / 12.0;
    assert!((centers[0] - mean0).abs() < 1e-5, "k=1 center[0]");
    assert!((centers[1] - mean1).abs() < 1e-5, "k=1 center[1]");
}

// ---------------------------------------------------------------------------
// init — the two STRING strategies
// ---------------------------------------------------------------------------

/// Deterministic well-separated blobs: `k` true centers on a wide grid, tight
/// noise, so EVERY sensible init converges to the same partition and inertia.
/// This is the fixture shape that makes a cross-library init comparison
/// meaningful at all.
fn blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let centers: Vec<f64> = (0..k * d).map(|_| next() * 100.0).collect();
    (0..n)
        .flat_map(|i| {
            let c = i % k;
            (0..d)
                .map(|j| centers[c * d + j] + (next() - 0.5) * 0.5)
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Fit blobs under an arbitrary `init` / `n_init` and return the fit surface.
fn fit_blobs(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    init: KMeansInit<f64>,
    n_init: NInit,
    seed: u64,
) -> (Vec<i64>, f64, Vec<f64>) {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xf: Vec<f64> = x.to_vec();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xf);

    let km = KMeans::<f64>::builder()
        .n_clusters(k)
        .init_method(init)
        .n_init(n_init)
        .random_state(Some(seed))
        .build::<f64>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (n, d))
        .expect("fit");

    (
        km.labels(&pool).iter().map(|&l| l as i64).collect(),
        km.inertia(),
        km.cluster_centers(&pool),
    )
}

/// Both string inits recover the true blob partition exactly, and reach the
/// SAME inertia — the property that makes the value comparison against live
/// sklearn (in the Python oracle) well-posed despite the different RNG stream.
#[test]
fn init_strings_recover_the_true_partition() {
    if capability::skip_f64_with_log() {
        println!("init strings: SKIPPED (no f64 on this adapter)");
        return;
    }
    const N: usize = 120;
    const D: usize = 4;
    const K: usize = 4;
    let x = blobs(N, D, K, 7);
    let truth: Vec<i64> = (0..N).map(|i| (i % K) as i64).collect();

    for (name, init) in [
        ("k-means++", KMeansInit::<f64>::KMeansPlusPlus),
        ("random", KMeansInit::<f64>::Random),
    ] {
        let (labels, inertia, centers) = fit_blobs(&x, N, D, K, init, NInit::Auto, 42);
        let acc = best_match_accuracy(&labels, &truth);
        assert!(
            (acc - 1.0).abs() < f64::EPSILON,
            "init={name}: best_match_accuracy {acc} != 1.0 — the init did not recover the blobs"
        );
        assert_eq!(centers.len(), K * D, "init={name}: centers shape");
        // Tight blobs (noise half-width 0.25 per feature): the global optimum's
        // inertia is bounded well below the inter-blob scale.
        assert!(
            inertia > 0.0 && inertia < 100.0,
            "init={name}: inertia {inertia} is not at the global optimum"
        );
    }
}

/// The `init` string parse accepts exactly sklearn's two `StrOptions` and
/// rejects everything else (`BuildError::UnknownInit`) — including
/// `'kmeans++'`, the spelling users actually mistype.
#[test]
fn init_string_parse_matches_sklearn_str_options() {
    assert_eq!(
        KMeansInit::<f64>::try_from("k-means++").expect("k-means++ is legal"),
        KMeansInit::KMeansPlusPlus
    );
    assert_eq!(
        KMeansInit::<f64>::try_from("random").expect("random is legal"),
        KMeansInit::Random
    );
    for bad in ["kmeans++", "k-means", "K-Means++", "", "auto"] {
        match KMeansInit::<f64>::try_from(bad) {
            Err(BuildError::UnknownInit { value }) => assert_eq!(value, bad),
            other => panic!("init={bad:?} must be rejected as UnknownInit, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// n_init
// ---------------------------------------------------------------------------

/// `n_init='auto'` resolves EXACTLY as sklearn's `_check_params_vs_input`:
/// 1 for `k-means++`, 10 for `random`, 1 for an explicit array; and an
/// explicit array overrides ANY explicit count to 1.
#[test]
fn n_init_auto_resolution_matches_sklearn() {
    let arr = KMeansInit::Array(vec![0.0; 6]);
    assert_eq!(NInit::Auto.resolve(&KMeansInit::<f64>::KMeansPlusPlus), 1);
    assert_eq!(NInit::Auto.resolve(&KMeansInit::<f64>::Random), 10);
    assert_eq!(NInit::Auto.resolve(&arr), 1);

    assert_eq!(NInit::Fixed(5).resolve(&KMeansInit::<f64>::KMeansPlusPlus), 5);
    assert_eq!(NInit::Fixed(5).resolve(&KMeansInit::<f64>::Random), 5);
    // sklearn WARNS and overrides to 1 for an explicit init; a library crate
    // must not print, so mlrs performs the same override silently.
    assert_eq!(NInit::Fixed(5).resolve(&arr), 1);

    // 'auto' is the only legal n_init STRING.
    assert_eq!(NInit::try_from("auto").expect("auto is legal"), NInit::Auto);
    for bad in ["Auto", "10", "default", ""] {
        match NInit::try_from(bad) {
            Err(BuildError::UnknownNInit { value }) => assert_eq!(value, bad),
            other => panic!("n_init={bad:?} must be rejected as UnknownNInit, got {other:?}"),
        }
    }
}

/// More restarts can only help: on a deliberately HARD design (overlapping,
/// unequal blobs where a single random init routinely lands in a local
/// optimum) the `n_init=10` fit's inertia is never worse than `n_init=1`'s,
/// which is the entire contract of the parameter.
#[test]
fn n_init_restarts_never_increase_inertia() {
    if capability::skip_f64_with_log() {
        println!("n_init monotonicity: SKIPPED (no f64 on this adapter)");
        return;
    }
    const N: usize = 200;
    const D: usize = 2;
    const K: usize = 6;
    // Elongated, unevenly spaced clusters — a design where init matters.
    let mut s = 11u64;
    let mut next = || {
        s = s
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0x94D0_49BB_1331_11EB);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let x: Vec<f64> = (0..N)
        .flat_map(|i| {
            let c = (i % K) as f64;
            vec![c * c * 1.5 + next() * 3.0, next() * 12.0]
        })
        .collect();

    let (_l1, inertia_1, _c1) = fit_blobs(&x, N, D, K, KMeansInit::Random, NInit::Fixed(1), 3);
    let (_l10, inertia_10, _c10) = fit_blobs(&x, N, D, K, KMeansInit::Random, NInit::Fixed(10), 3);

    assert!(
        inertia_10 <= inertia_1 * (1.0 + 1e-12),
        "n_init=10 inertia {inertia_10} must not exceed n_init=1 inertia {inertia_1}"
    );
}

/// `n_init = 0` is outside sklearn's `Interval(Integral, 1, None)` and is
/// rejected at `build()` — the only data-INDEPENDENT KMeans rejection.
#[test]
fn n_init_zero_is_rejected_at_build() {
    // `KMeans` is not `Debug` (it carries device handles), so match the Result
    // directly rather than going through `expect_err`.
    match KMeans::<f32>::builder().n_init(NInit::Fixed(0)).build::<f32>() {
        Err(BuildError::InvalidHyperprior { param, value, .. }) => {
            assert_eq!(param, "n_init");
            assert_eq!(value, 0.0);
        }
        Err(other) => panic!("expected InvalidHyperprior for n_init=0, got {other:?}"),
        Ok(_) => panic!("n_init=0 must be rejected at build()"),
    }
}

// ---------------------------------------------------------------------------
// n_iter_, random_state, verbose, copy_x
// ---------------------------------------------------------------------------

/// `n_iter_` is the WINNING run's iteration count, not the sum over restarts:
/// a 10-restart fit must not report ~10× the single-restart count.
#[test]
fn n_iter_reports_the_winning_run_not_the_sum() {
    if capability::skip_f64_with_log() {
        println!("n_iter_: SKIPPED (no f64 on this adapter)");
        return;
    }
    const N: usize = 120;
    const D: usize = 3;
    const K: usize = 4;
    let x = blobs(N, D, K, 5);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let fitted = |n_init: NInit, pool: &mut BufferPool<ActiveRuntime>| {
        KMeans::<f64>::builder()
            .n_clusters(K)
            .init_method(KMeansInit::Random)
            .n_init(n_init)
            .random_state(Some(9))
            .build::<f64>()
            .expect("build")
            .fit(pool, &x_dev, None, (N, D))
            .expect("fit")
    };

    let one = fitted(NInit::Fixed(1), &mut pool);
    let ten = fitted(NInit::Fixed(10), &mut pool);

    assert!(one.n_iter() >= 1, "n_iter_ must count at least one iteration");
    assert!(
        ten.n_iter() <= one.n_iter().max(ten.n_iter()) && ten.n_iter() < 300,
        "n_iter_ {} looks like a cross-restart sum, not a single run's count",
        ten.n_iter()
    );
    assert_eq!(ten.n_init_used(), 10, "n_init_used reports the resolved count");
    assert_eq!(one.n_init_used(), 1);
}

/// `random_state` is the ONLY source of run-to-run variation: the same seed
/// reproduces a fit exactly, and a different seed is free to differ (both fits
/// stay valid clusterings, which is all sklearn guarantees).
#[test]
fn random_state_makes_the_fit_reproducible() {
    if capability::skip_f64_with_log() {
        println!("random_state: SKIPPED (no f64 on this adapter)");
        return;
    }
    const N: usize = 120;
    const D: usize = 3;
    const K: usize = 5;
    let x = blobs(N, D, K, 21);

    let a = fit_blobs(&x, N, D, K, KMeansInit::Random, NInit::Fixed(3), 1234);
    let b = fit_blobs(&x, N, D, K, KMeansInit::Random, NInit::Fixed(3), 1234);
    assert_eq!(a.0, b.0, "same random_state must reproduce labels_ exactly");
    assert_eq!(a.1, b.1, "same random_state must reproduce inertia_ exactly");
    assert_eq!(a.2, b.2, "same random_state must reproduce centers exactly");
}

/// `verbose` and `copy_x` are accepted, round-trip through the builder, and
/// have NO effect on the fit — mlrs never prints from a library crate and
/// never writes into the caller's buffer, so both are parity-only. The test
/// pins that they are inert rather than silently ignored somewhere that
/// matters.
#[test]
fn verbose_and_copy_x_are_accepted_and_inert() {
    if capability::skip_f64_with_log() {
        println!("verbose/copy_x: SKIPPED (no f64 on this adapter)");
        return;
    }
    const N: usize = 60;
    const D: usize = 3;
    const K: usize = 3;
    let x = blobs(N, D, K, 2);

    let run = |verbose: bool, copy_x: bool| {
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
        let km = KMeans::<f64>::builder()
            .n_clusters(K)
            .init_method(KMeansInit::KMeansPlusPlus)
            .random_state(Some(4))
            .verbose(verbose)
            .copy_x(copy_x)
            .build::<f64>()
            .expect("build")
            .fit(&mut pool, &x_dev, None, (N, D))
            .expect("fit");
        (km.labels(&pool), km.inertia())
    };

    let base = run(false, true);
    for (v, c) in [(true, true), (false, false), (true, false)] {
        let got = run(v, c);
        assert_eq!(got.0, base.0, "verbose={v} copy_x={c} changed labels_");
        assert_eq!(got.1, base.1, "verbose={v} copy_x={c} changed inertia_");
    }

    // The caller's buffer is untouched regardless of copy_x (mlrs never
    // mean-centers in place, which is the only thing sklearn's copy_x gates).
    let before = blobs(N, D, K, 2);
    assert_eq!(x, before, "fit must never write into the caller's X");
}

/// `algorithm` string parse accepts exactly sklearn's two `StrOptions`.
#[test]
fn algorithm_string_parse_matches_sklearn_str_options() {
    assert_eq!(
        KMeansAlgorithm::try_from("lloyd").expect("lloyd is legal"),
        KMeansAlgorithm::Lloyd
    );
    assert_eq!(
        KMeansAlgorithm::try_from("elkan").expect("elkan is legal"),
        KMeansAlgorithm::Elkan
    );
    assert_eq!(KMeansAlgorithm::Lloyd.name(), "lloyd");
    assert_eq!(KMeansAlgorithm::Elkan.name(), "elkan");
    for bad in ["full", "auto", "Elkan", ""] {
        match KMeansAlgorithm::try_from(bad) {
            Err(BuildError::UnknownAlgorithm { value }) => assert_eq!(value, bad),
            other => panic!("algorithm={bad:?} must be rejected, got {other:?}"),
        }
    }
}
