//! `SpectralClustering` (SPECTRAL-02) sklearn oracle + parameter-surface tests.
//!
//! The oracle cases load the committed WELL-SEPARATED fixture, fit
//! `SpectralClustering` (rbf affinity → normalized Laplacian → smallest
//! `n_components` eigenvectors with `drop_first = FALSE` → `/dd` recovery →
//! k-means), and assert `labels_` matches sklearn EXACTLY up to a label
//! permutation (`mlrs_core::best_match_accuracy == 1.0`) — no tolerance band
//! (labels are integers; they match or they don't).
//!
//! The well-separated fixture makes the partition UNIQUE up to permutation, so
//! the SplitMix64-vs-MT19937 k-means++ RNG gap is immaterial: both converge to
//! the same labeling.
//!
//! The remaining cases cover the SPECTRAL-PERF-CPU rewrite: the `n_samples > 64`
//! cap is gone, all three `assign_labels` strategies land the same partition on
//! a well-separated graph, the whole `pairwise_kernels` affinity family is
//! reachable, and the data-independent parameter bounds reject at `build()`.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::SpectralClustering;
use mlrs_algos::error::AlgoError;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase};

/// SpectralClustering fixture geometry (gen_oracle.py `SC_N_SAMPLES` ×
/// `SC_N_FEATURES`, `SC_N_CLUSTERS` clusters).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 2;
const N_CLUSTERS: usize = 3;

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
        _ => unreachable!("spectral_clustering fixtures are f32/f64 only"),
    }
}

/// Read the reference `labels_` (stored as f64 in the `.npz`) into an `i64` slice
/// for the `best_match_accuracy` label-permutation compare.
fn ref_labels(case: &OracleCase) -> Vec<i64> {
    case.expect_f64("labels").iter().map(|&v| v as i64).collect()
}

/// A pure counter-based hash (the SplitMix64 finalizer) mapped to `[-0.5, 0.5)`.
/// Used instead of a stateful PRNG so a blob's coordinates depend only on the
/// point index — the fixture is byte-reproducible and no test shares a stream.
fn jitter(i: u64) -> f64 {
    let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Three well-separated 2-D blobs of `per` points each. Returns the row-major
/// `3·per × 2` design and the true block labels.
///
/// Each blob is an ISOTROPIC cloud of radius `0.5` around its center, and the
/// centers are 6 apart. Isotropy is load-bearing for the `nearest_neighbors`
/// case: a blob laid out along a line (or with any discontinuity in the point
/// ordering) has a kNN graph that fragments into several components, which
/// silently changes what the smallest `n_components` eigenvectors span — the
/// embedding then resolves the FRAGMENTS rather than the blobs. A dense 2-D
/// cloud at `k = 10` is comfortably connected.
///
/// With `gamma = 0.5` the within-blob rbf affinity is `~exp(-0.25)` and the
/// between-blob one `~exp(-18)`, so the graph is three near-disconnected
/// components joined by numerically tiny edges. That is the regime where the
/// partition is UNIQUE, which is what lets the three `assign_labels` strategies
/// be compared against each other.
fn blobs(per: usize) -> (Vec<f64>, Vec<i64>) {
    let centers = [(0.0f64, 0.0f64), (6.0, 0.0), (3.0, 6.0)];
    let mut x = Vec::with_capacity(3 * per * 2);
    let mut y = Vec::with_capacity(3 * per);
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for i in 0..per {
            let seed = (c * 100_003 + i) as u64 + 1;
            x.push(cx + jitter(seed));
            x.push(cy + jitter(seed.wrapping_add(0x5DEE_CE66)));
            y.push(c as i64);
        }
    }
    (x, y)
}

/// Fit a `SpectralClustering` (own default constructor: `affinity="rbf"`,
/// `gamma=1.0`; `n_components=None → n_clusters`) on the fixture's `X` and return
/// the host `labels_` as `i64` for the permutation compare.
fn fit_labels<F>(case: &OracleCase) -> Vec<i64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    // sklearn's own SpectralClustering defaults: rbf affinity, gamma=1.0 literal,
    // n_components=None → n_clusters. The seed is immaterial on the
    // well-separated fixture.
    let sc = SpectralClustering::<F>::builder()
        .n_clusters(N_CLUSTERS)
        .n_components(None)
        .affinity("rbf".to_string())
        .gamma(1.0)
        .n_neighbors(10)
        .seed(42)
        .build::<F>()
        .expect("SpectralClustering build with valid hyperparameters");
    let sc = sc
        .fit(&mut pool, &x_dev, None, (N_SAMPLES, N_FEATURES))
        .expect("SpectralClustering::fit on a valid shape");

    sc.labels(&pool).iter().map(|&l| l as i64).collect()
}

/// SPECTRAL-02: `labels_` matches sklearn EXACTLY up to a label permutation on the
/// well-separated fixture, f64 strict. Gated by `skip_f64_with_log`.
#[test]
fn spectral_clustering() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_clustering f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_clustering_f64_seed42.npz"))
        .expect("load spectral_clustering_f64");
    let labels_ref = ref_labels(&case);
    assert_eq!(labels_ref.len(), N_SAMPLES, "reference labels are length n");

    let labels = fit_labels::<f64>(&case);
    let acc = best_match_accuracy(&labels, &labels_ref);
    println!("spectral_clustering f64 best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "spectral_clustering f64: best_match_accuracy {acc} != 1.0 \
         (labels are not a permutation of sklearn's on the well-separated fixture)"
    );
}

/// SPECTRAL-02 (f32): `labels_` EXACT up to permutation (labels are integers — no
/// documented band, the exact-labels gate holds at f32 too).
#[test]
fn spectral_clustering_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("spectral_clustering_f32_seed42.npz"))
        .expect("load spectral_clustering_f32");
    let labels_ref = ref_labels(&case);
    assert_eq!(labels_ref.len(), N_SAMPLES, "reference labels are length n");

    let labels = fit_labels::<f32>(&case);
    let acc = best_match_accuracy(&labels, &labels_ref);
    println!("spectral_clustering f32 best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "spectral_clustering f32: best_match_accuracy {acc} != 1.0 \
         (labels are not a permutation of sklearn's on the well-separated fixture)"
    );
}

/// REPLACES the former `reject_oversize`: `n_samples = 65` used to be rejected
/// with `AlgoError::NSamplesExceedsMaxDim`, because the dense cyclic-Jacobi `eig`
/// kernel stages `MAX_DIM x MAX_DIM` shared memory and so capped `n <= 64`. The
/// host pipeline has no such cap, and this test is the live proof: a fit at
/// `n = 600` — an order of magnitude past the old ceiling — must SUCCEED and
/// return a valid partition.
///
/// The geometry is the three-blob fixture, whose partition is unique, so the
/// labels are checked against the true blocks and not merely for finiteness.
#[test]
fn large_n_is_no_longer_capped() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 200usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .gamma(0.5)
        .random_state(Some(7))
        .build::<f64>()
        .expect("SpectralClustering build");
    let fitted = sc
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("fit(n = 600) must succeed now that the MAX_DIM cap is gone");

    let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
    assert_eq!(labels.len(), n, "labels_ must be length n");
    assert_eq!(fitted.n_features_in(), 2, "n_features_in_");
    assert_eq!(fitted.n_samples(), n, "n_samples");
    // A dense rbf affinity has no exact zeros, so the graph is one component.
    assert_eq!(fitted.n_graph_components(), 1, "rbf graphs are connected");
    assert_eq!(
        fitted.affinity_matrix_dense().len(),
        n * n,
        "affinity_matrix_ densifies to n x n"
    );
    assert!(
        fitted.affinity_matrix_sparse().is_none(),
        "a kernel affinity is DENSE, as it is in sklearn"
    );
    let acc = best_match_accuracy(&labels, &truth);
    println!("large_n_is_no_longer_capped (n = {n}) best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "the three blobs must be recovered exactly, got accuracy {acc}"
    );
}

/// All three `assign_labels` strategies (`kmeans` / `discretize` / `cluster_qr`)
/// must recover the same unique partition on a well-separated graph.
///
/// This is the strongest statement available without pinning sklearn's exact
/// local optimum: `discretize` and `cluster_qr` are DIFFERENT algorithms from
/// k-means (a rotation search and a pivoted QR), so their agreeing here means
/// each one is finding the true block structure rather than an artifact.
#[test]
fn assign_labels_strategies_agree() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 40usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    for strategy in ["kmeans", "discretize", "cluster_qr"] {
        let sc = SpectralClustering::<f64>::builder()
            .n_clusters(3)
            .gamma(0.5)
            .assign_labels(strategy.to_string())
            .random_state(Some(1))
            .build::<f64>()
            .expect("SpectralClustering build");
        let fitted = sc
            .fit(&mut pool, &x_dev, None, (n, 2))
            .expect("fit with a valid assign_labels");
        let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
        assert!(
            labels.iter().all(|&l| (0..3).contains(&l)),
            "{strategy}: labels must lie in 0..n_clusters"
        );
        let acc = best_match_accuracy(&labels, &truth);
        println!("assign_labels={strategy} best_match_accuracy = {acc}");
        assert!(
            (acc - 1.0).abs() < 1e-12,
            "assign_labels={strategy}: expected the true 3-block partition, got {acc}"
        );
    }
}

/// `n_init` is honored: a single restart and ten restarts must both land the
/// unique partition, and `n_init = 0` is rejected at `build()` (sklearn's
/// `Interval(Integral, 1, None, closed="left")`).
#[test]
fn n_init_restarts() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 30usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    for n_init in [1usize, 10] {
        let sc = SpectralClustering::<f64>::builder()
            .n_clusters(3)
            .gamma(0.5)
            .n_init(n_init)
            .random_state(Some(3))
            .build::<f64>()
            .expect("SpectralClustering build");
        let fitted = sc
            .fit(&mut pool, &x_dev, None, (n, 2))
            .expect("fit with a valid n_init");
        let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
        let acc = best_match_accuracy(&labels, &truth);
        assert!(
            (acc - 1.0).abs() < 1e-12,
            "n_init = {n_init}: expected the true partition, got {acc}"
        );
    }

    let err = SpectralClustering::<f64>::builder()
        .n_init(0)
        .build::<f64>()
        .map(|_| ())
        .expect_err("n_init = 0 must be rejected at build()");
    let msg = err.to_string();
    assert!(
        msg.contains("n_init") && msg.contains(">= 1"),
        "the n_init rejection must name the parameter and the bound: {msg}"
    );
}

/// The non-`rbf` `pairwise_kernels` family — which only `SpectralClustering`
/// reaches — is wired through end to end.
///
/// `laplacian` (`exp(-γ‖x−y‖₁)`) is a genuine similarity on this geometry, so it
/// must recover the true partition. The non-negative rest are only required to
/// FIT and return in-range labels: `linear` / `poly` / `sigmoid` / `chi2` are
/// not similarity kernels on arbitrary data (sklearn documents that and does not
/// check it either), so asserting a particular partition for them would be
/// asserting a property the kernel does not have.
///
/// `additive_chi2` is the one member that is NEGATIVE by construction (it is a
/// negated distance), which makes every degree negative and the normalized
/// Laplacian undefined. sklearn hands the resulting NaN matrix to ARPACK; mlrs
/// rejects it with a typed error, which this pins.
#[test]
fn kernel_affinity_family() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 30usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    // chi2 / additive_chi2 require non-negative inputs; shift the design so every
    // kernel in the family sees data it is defined on.
    let x_pos: Vec<f64> = x.iter().map(|&v| v + 2.0).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_pos);

    // The similarity kernel: must find the blocks.
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("laplacian".to_string())
        .gamma(0.5)
        .random_state(Some(11))
        .build::<f64>()
        .expect("SpectralClustering build");
    let fitted = sc
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("laplacian-kernel affinity fit");
    let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
    let acc = best_match_accuracy(&labels, &truth);
    println!("affinity=laplacian best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "the laplacian kernel must recover the blocks, got {acc}"
    );

    for affinity in ["linear", "poly", "polynomial", "sigmoid", "cosine", "chi2"] {
        let sc = SpectralClustering::<f64>::builder()
            .n_clusters(3)
            .affinity(affinity.to_string())
            .gamma(0.5)
            .degree(2.0)
            .coef0(1.0)
            .random_state(Some(11))
            .build::<f64>()
            .expect("SpectralClustering build");
        let fitted = sc
            .fit(&mut pool, &x_dev, None, (n, 2))
            .unwrap_or_else(|e| panic!("affinity '{affinity}' must be supported: {e}"));
        let labels = fitted.labels(&pool);
        assert_eq!(labels.len(), n, "affinity '{affinity}': labels length");
        assert!(
            labels.iter().all(|&l| (0..3).contains(&l)),
            "affinity '{affinity}': labels must lie in 0..n_clusters"
        );
    }

    // `additive_chi2` is a NEGATED distance: every off-diagonal entry is <= 0,
    // so every degree is negative and `dd = sqrt(deg)` is NaN. Rejected, not
    // silently propagated.
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("additive_chi2".to_string())
        .build::<f64>()
        .expect("SpectralClustering build");
    match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
        Err(AlgoError::InvalidGraphInput { estimator, reason }) => {
            assert_eq!(estimator, "spectral_clustering");
            assert!(
                reason.contains("non-finite degree"),
                "the rejection must name the violated invariant: {reason}"
            );
        }
        other => panic!("expected InvalidGraphInput for additive_chi2, got {other:?}"),
    }

    // An unknown affinity string is a typed rejection, not a silent fall-through.
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("not_a_kernel".to_string())
        .build::<f64>()
        .expect("SpectralClustering build");
    match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
        Err(AlgoError::InvalidKernel { estimator, kernel }) => {
            assert_eq!(estimator, "spectral_clustering");
            assert_eq!(kernel, "not_a_kernel");
        }
        other => panic!("expected InvalidKernel, got {other:?}"),
    }
}

/// The `nearest_neighbors` affinity keeps its graph SPARSE and uses sklearn's
/// `SpectralClustering` default of `n_neighbors = 10` — NOT `SpectralEmbedding`'s
/// `max(n_samples // 10, 1)`, which at `n = 200` would be 20 and would build a
/// visibly different graph.
#[test]
fn nearest_neighbors_affinity_is_sparse_and_uses_the_int_default() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 100usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("nearest_neighbors".to_string())
        .random_state(Some(5))
        .build::<f64>()
        .expect("SpectralClustering build");
    let fitted = sc
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("nearest_neighbors affinity fit");

    let csr = fitted
        .affinity_matrix_sparse()
        .expect("the kNN affinity must stay SPARSE");
    assert_eq!(csr.indptr.len(), n + 1, "CSR indptr is n + 1");
    let max_row = (0..n)
        .map(|i| (csr.indptr[i + 1] - csr.indptr[i]) as usize)
        .max()
        .expect("n > 0");
    // Row `i` holds its own `k = 10` outgoing edges plus whatever incoming ones
    // the symmetrization adds; with `n_neighbors = 20` the FLOOR alone would
    // already exceed 19.
    assert!(
        (10..=19).contains(&max_row),
        "expected a k = 10 connectivity graph, max row nnz = {max_row}"
    );

    let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
    let acc = best_match_accuracy(&labels, &truth);
    println!("affinity=nearest_neighbors best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "the kNN graph must recover the blocks, got {acc}"
    );
    // Three tight blobs at k = 10 give three disconnected kNN components.
    assert_eq!(
        fitted.n_graph_components(),
        3,
        "the kNN graph over three separated blobs has three components"
    );
}

/// `affinity = "precomputed"` consumes an `n × n` matrix rather than an
/// `n × n_features` design, and `n_features_in_` is then `n_samples` — the same
/// thing sklearn reports, because it validates the square matrix as `X`.
#[test]
fn precomputed_affinity() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 20usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    // Build the rbf affinity by hand and hand it over precomputed.
    let mut aff = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = x[i * 2] - x[j * 2];
            let dy = x[i * 2 + 1] - x[j * 2 + 1];
            aff[i * n + j] = (-0.5 * (dx * dx + dy * dy)).exp();
        }
    }
    let aff_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &aff);

    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("precomputed".to_string())
        .random_state(Some(2))
        .build::<f64>()
        .expect("SpectralClustering build");
    let fitted = sc
        .fit(&mut pool, &aff_dev, None, (n, n))
        .expect("precomputed affinity fit");
    assert_eq!(fitted.n_features_in(), n, "n_features_in_ is n for precomputed");
    let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
    let acc = best_match_accuracy(&labels, &truth);
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "the precomputed rbf affinity must reproduce the rbf partition, got {acc}"
    );

    // A precomputed affinity handed a NON-square operand is rejected before any
    // indexing (ASVS V5).
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("precomputed".to_string())
        .build::<f64>()
        .expect("SpectralClustering build");
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
        Err(AlgoError::InvalidGraphInput { estimator, .. }) => {
            assert_eq!(estimator, "spectral_clustering");
        }
        other => panic!("expected InvalidGraphInput for a non-square precomputed X, got {other:?}"),
    }
}

/// `fit_from_host_slice` (the no-upload arm) must produce byte-identical labels
/// to the `DeviceArray` `Fit::fit` entry point — they share `fit_host_core`, and
/// this pins that they cannot drift.
#[test]
fn host_slice_entry_point_matches_device_entry_point() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 25usize;
    let (x, _) = blobs(per);
    let n = 3 * per;

    let unfit = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .gamma(0.5)
        .random_state(Some(9))
        .build::<f64>()
        .expect("SpectralClustering build");
    assert!(
        unfit.host_fit_applicable((n, 2)),
        "the spectral pipeline is host-side on every backend"
    );
    let via_host = unfit
        .fit_from_host_slice(&mut pool, &x, (n, 2))
        .expect("fit_from_host_slice");

    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let via_device = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .gamma(0.5)
        .random_state(Some(9))
        .build::<f64>()
        .expect("SpectralClustering build")
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("Fit::fit");

    assert_eq!(
        via_host.labels(&pool),
        via_device.labels(&pool),
        "the host-slice and device entry points must agree exactly"
    );
}

/// `fit_predict` returns the fitted `labels_` as an independent device buffer.
#[test]
fn fit_predict_returns_the_fitted_labels() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 20usize;
    let (x, _) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let (fitted, labels_dev) = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .gamma(0.5)
        .random_state(Some(4))
        .build::<f64>()
        .expect("SpectralClustering build")
        .fit_predict(&mut pool, &x_dev, (n, 2))
        .expect("fit_predict");
    assert_eq!(
        labels_dev.to_host(&pool),
        fitted.labels(&pool),
        "fit_predict must return the fitted labels_"
    );
}

/// The string-valued parameters are validated at `fit`, where sklearn's
/// `_fit_context` `StrOptions` validation rejects them, and `gamma = 0` is
/// ACCEPTED — sklearn 1.9's constraint is `Interval(Real, 0, None,
/// closed="left")`, so zero is in range (it yields a constant all-ones affinity).
#[test]
fn string_parameter_validation_and_gamma_zero() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 10usize;
    let (x, _) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    for (assign, solver) in [("kmeans", Some("elephant")), ("wolves", None)] {
        let sc = SpectralClustering::<f64>::builder()
            .n_clusters(3)
            .assign_labels(assign.to_string())
            .eigen_solver(solver.map(str::to_string))
            .build::<f64>()
            .expect("SpectralClustering build (string params validate at fit)");
        match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
            Err(AlgoError::Unsupported { estimator, .. }) => {
                assert_eq!(estimator, "spectral_clustering");
            }
            other => panic!("expected Unsupported for ({assign}, {solver:?}), got {other:?}"),
        }
    }

    // Every accepted `eigen_solver` reaches the same solver and the same labels.
    let mut reference: Option<Vec<i32>> = None;
    for solver in [None, Some("arpack"), Some("lobpcg"), Some("amg")] {
        let sc = SpectralClustering::<f64>::builder()
            .n_clusters(3)
            .gamma(0.5)
            .eigen_solver(solver.map(str::to_string))
            .random_state(Some(6))
            .build::<f64>()
            .expect("SpectralClustering build");
        let fitted = sc
            .fit(&mut pool, &x_dev, None, (n, 2))
            .expect("every accepted eigen_solver routes to the one solver");
        let labels = fitted.labels(&pool);
        match &reference {
            None => reference = Some(labels),
            Some(r) => assert_eq!(r, &labels, "eigen_solver = {solver:?} changed the labels"),
        }
    }

    // gamma = 0 is legal (all-ones affinity), NOT an InvalidGamma.
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(2)
        .gamma(0.0)
        .build::<f64>()
        .expect("SpectralClustering build");
    let fitted = sc
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("gamma = 0 is inside sklearn 1.9's closed='left' interval");
    assert_eq!(fitted.labels(&pool).len(), n);

    // A negative gamma is not.
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(2)
        .gamma(-1.0)
        .build::<f64>()
        .expect("SpectralClustering build");
    match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
        Err(AlgoError::InvalidGamma { estimator, gamma }) => {
            assert_eq!(estimator, "spectral_clustering");
            assert_eq!(gamma, -1.0);
        }
        other => panic!("expected InvalidGamma for gamma = -1, got {other:?}"),
    }
}

/// The data-INDEPENDENT hyperparameter bounds reject at `build()`, and the
/// data-DEPENDENT `n_clusters <= n_samples` at `fit`.
#[test]
fn numeric_parameter_bounds() {
    let _ = env_logger::builder().is_test(true).try_init();
    assert!(
        SpectralClustering::<f64>::builder()
            .n_clusters(0)
            .build::<f64>()
            .is_err(),
        "n_clusters = 0 must be rejected at build()"
    );
    assert!(
        SpectralClustering::<f64>::builder()
            .n_components(Some(0))
            .build::<f64>()
            .is_err(),
        "n_components = 0 must be rejected at build()"
    );
    assert!(
        SpectralClustering::<f64>::builder()
            .n_neighbors(0)
            .build::<f64>()
            .is_err(),
        "n_neighbors = 0 must be rejected at build()"
    );
    assert!(
        SpectralClustering::<f64>::builder()
            .eigen_tol(Some(-1.0))
            .build::<f64>()
            .is_err(),
        "a negative eigen_tol must be rejected at build()"
    );
    assert!(
        SpectralClustering::<f64>::builder()
            .degree(-1.0)
            .build::<f64>()
            .is_err(),
        "a negative degree must be rejected at build()"
    );
    assert!(
        SpectralClustering::<f64>::builder()
            .coef0(f64::NAN)
            .build::<f64>()
            .is_err(),
        "a non-finite coef0 must be rejected at build()"
    );
    // eigen_tol = 0 is sklearn's `closed="left"` lower bound and is legal.
    assert!(
        SpectralClustering::<f64>::builder()
            .eigen_tol(Some(0.0))
            .build::<f64>()
            .is_ok(),
        "eigen_tol = 0 is inside the interval"
    );

    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, _) = blobs(4);
    let n = 12usize;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(n + 1)
        .build::<f64>()
        .expect("n_clusters > n_samples is data-DEPENDENT, so build() accepts it");
    match sc.fit(&mut pool, &x_dev, None, (n, 2)).map(|_| ()) {
        Err(AlgoError::InvalidK {
            estimator,
            k,
            n_samples,
        }) => {
            assert_eq!(estimator, "spectral_clustering");
            assert_eq!(k, n + 1);
            assert_eq!(n_samples, n);
        }
        other => panic!("expected InvalidK for n_clusters > n_samples, got {other:?}"),
    }
}

/// BLDR-01: `SpectralClustering::new()` (the single-source defaults) equals
/// `SpectralClustering::builder().build()` (the builder defaults re-derived from
/// `new`). The builder must round-trip every one of the sklearn parameters.
#[test]
fn spectral_clustering_defaults_equal() {
    let from_new = SpectralClustering::<f32>::new();
    let from_builder = SpectralClustering::<f32>::builder()
        .build::<f32>()
        .expect("default SpectralClustering builder build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "SpectralClustering::new() must equal SpectralClustering::builder().build() (BLDR-01)"
    );
    // A changed parameter must break the equality, i.e. `hyperparams_eq` really
    // compares the whole surface rather than a stale subset.
    for changed in [
        SpectralClustering::<f32>::builder().n_clusters(2).build::<f32>(),
        SpectralClustering::<f32>::builder()
            .eigen_solver(Some("amg".to_string()))
            .build::<f32>(),
        SpectralClustering::<f32>::builder()
            .n_components(Some(3))
            .build::<f32>(),
        SpectralClustering::<f32>::builder()
            .random_state(Some(1))
            .build::<f32>(),
        SpectralClustering::<f32>::builder().n_init(3).build::<f32>(),
        SpectralClustering::<f32>::builder().gamma(2.0).build::<f32>(),
        SpectralClustering::<f32>::builder()
            .affinity("cosine".to_string())
            .build::<f32>(),
        SpectralClustering::<f32>::builder().n_neighbors(3).build::<f32>(),
        SpectralClustering::<f32>::builder()
            .eigen_tol(Some(1e-3))
            .build::<f32>(),
        SpectralClustering::<f32>::builder()
            .assign_labels("cluster_qr".to_string())
            .build::<f32>(),
        SpectralClustering::<f32>::builder().degree(4.0).build::<f32>(),
        SpectralClustering::<f32>::builder().coef0(0.0).build::<f32>(),
        SpectralClustering::<f32>::builder().n_jobs(Some(-1)).build::<f32>(),
        SpectralClustering::<f32>::builder().verbose(true).build::<f32>(),
    ] {
        let changed = changed.expect("each single-parameter change is valid");
        assert!(
            !from_new.hyperparams_eq(&changed),
            "hyperparams_eq must observe every parameter"
        );
    }
}

/// A DISCONNECTED affinity graph is the case a single-vector Krylov solver gets
/// silently wrong, and `SpectralClustering` walks into it by design.
///
/// The normalized Laplacian of a graph with `c` connected components has
/// eigenvalue `0` with multiplicity exactly `c`, and its eigenspace is spanned
/// by the per-component degree vectors — so after the `/dd` recovery every
/// embedding column is CONSTANT within a component. A Krylov space built from
/// one starting vector contains only ONE direction from that eigenspace, so a
/// thick-restart Lanczos returns one such vector and fills the rest from the
/// next DISTINCT eigenvalues: genuine eigenpairs with tiny residuals, which pass
/// the convergence test while being the wrong answer. `spectral_host::run`
/// therefore routes a disconnected graph to the dense arm.
///
/// This pins the OUTCOME (piecewise-constant columns, hence the exact
/// partition) at `n = 300`, an order of magnitude past the dense-arm crossover,
/// so a future block-Krylov rewrite is free to change the ROUTE.
#[test]
fn disconnected_graph_resolves_the_degenerate_null_space() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let per = 100usize;
    let (x, truth) = blobs(per);
    let n = 3 * per;
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let fitted = SpectralClustering::<f64>::builder()
        .n_clusters(3)
        .affinity("nearest_neighbors".to_string())
        .n_neighbors(10)
        .random_state(Some(5))
        .build::<f64>()
        .expect("SpectralClustering build")
        .fit(&mut pool, &x_dev, None, (n, 2))
        .expect("fit on a disconnected kNN graph");

    assert_eq!(
        fitted.n_graph_components(),
        3,
        "the fixture must actually be disconnected, or this test proves nothing"
    );
    let labels: Vec<i64> = fitted.labels(&pool).iter().map(|&l| l as i64).collect();
    let acc = best_match_accuracy(&labels, &truth);
    println!("disconnected-graph best_match_accuracy = {acc}");
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "a disconnected graph's null space must be resolved in FULL — a partial \
         basis leaves the embedding non-constant inside a component and the \
         partition wrong (got accuracy {acc})"
    );
}
