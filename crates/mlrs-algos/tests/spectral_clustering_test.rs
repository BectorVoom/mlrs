//! Plan 09-04 — SpectralClustering (SPECTRAL-02) sklearn oracle test.
//!
//! Activated from the 09-01 Nyquist `#[ignore]` scaffold: loads the committed
//! WELL-SEPARATED fixture (D-10), fits the device `SpectralClustering` (rbf
//! affinity → normalized Laplacian → v1 `eig` → `/dd` recovery with
//! `drop_first = FALSE`, D-11 → v1 `KMeans::new`), and asserts `labels_` matches
//! sklearn EXACTLY up to a label permutation
//! (`mlrs_core::best_match_accuracy == 1.0`) — no tolerance band (labels are
//! integers; they match or they don't, D-10).
//!
//! The well-separated fixture makes the partition UNIQUE up to permutation, so
//! the SplitMix64-vs-MT19937 init RNG gap between `KMeans::new` and sklearn is
//! immaterial: both converge to the same labeling (D-10, the spectral analogue of
//! the Phase-5 DBSCAN tuned-fixture design).
//!
//! f64 carries the `skip_f64_with_log` capability gate verbatim; f32 is ALSO
//! exact (the labels are integers, not floats — no documented band).
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

/// Fit a `SpectralClustering` (own default constructor: `affinity="rbf"`,
/// `gamma=1.0` D-01/D-04; `n_components=None → n_clusters` D-11) on the fixture's
/// `X` and return the host `labels_` as `i64` for the permutation compare.
fn fit_labels<F>(case: &OracleCase) -> Vec<i64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    // sklearn's own SpectralClustering defaults: rbf affinity, gamma=1.0 literal
    // (D-04), n_components=None → n_clusters (D-11). seed immaterial on the
    // well-separated fixture (D-10). The wide builder folds the 6-arg legacy new.
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
/// well-separated fixture (D-10), f64 strict. Gated by `skip_f64_with_log`.
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
/// documented band, the exact-labels gate holds at f32 too, D-10).
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

/// `n_samples > 64` is rejected with `AlgoError::NSamplesExceedsMaxDim` BEFORE any
/// device work (D-06). A live `fit(n=65)` must return the typed spectral-cap error
/// without any affinity / Laplacian / eig / KMeans launch.
#[test]
fn reject_oversize() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 65usize;
    let d = 3usize;
    let x_host: Vec<f64> = vec![0.0; n * d];
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let sc = SpectralClustering::<f64>::builder()
        .n_clusters(N_CLUSTERS)
        .n_components(None)
        .affinity("rbf".to_string())
        .gamma(1.0)
        .n_neighbors(10)
        .seed(42)
        .build::<f64>()
        .expect("SpectralClustering build with valid hyperparameters");
    let err = match sc.fit(&mut pool, &x_dev, None, (n, d)) {
        Ok(_) => panic!("fit(n=65) must reject before any device work"),
        Err(e) => e,
    };

    let msg = err.to_string();
    match err {
        AlgoError::NSamplesExceedsMaxDim {
            estimator,
            n_samples,
            max,
        } => {
            assert_eq!(estimator, "spectral_clustering");
            assert_eq!(n_samples, 65);
            assert_eq!(max, 64);
            assert!(
                msg.contains("65") && msg.contains("64") && msg.contains("MAX_DIM"),
                "NSamplesExceedsMaxDim message must name n_samples + the cap: {msg}"
            );
        }
        other => panic!("expected NSamplesExceedsMaxDim, got {other:?}"),
    }
}

/// BLDR-01: `SpectralClustering::new()` (the single-source defaults) equals
/// `SpectralClustering::builder().build()` (the builder defaults re-derived from
/// `new`). The wide 6-field builder must round-trip the defaults exactly.
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
}
