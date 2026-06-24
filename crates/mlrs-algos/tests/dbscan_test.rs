//! Plan 05-07 — DBSCAN (CLUSTER-02) sklearn oracle tests.
//!
//! Activated from the 05-01 Nyquist `#[ignore]` scaffold: each function loads
//! the committed `dbscan_{f32,f64}_seed42.npz` fixture (which carries `eps`,
//! `min_samples`, the sklearn `labels` with the `-1` noise sentinel, and
//! `core_sample_indices`), constructs `DBSCAN::new(eps, min_samples)`, fits via
//! the device eps-core mask + the host index-ordered DFS, and asserts:
//!   - `core_sample_indices_` matches the sklearn fixture set EXACTLY (an integer
//!     set — no tolerance; it is a count threshold), and
//!   - `labels_` (noise = `-1`) matches sklearn up to a cluster-label permutation
//!     (`mlrs_core::best_match_accuracy == 1.0`, D-09) — the `-1` noise label is
//!     carried through the permutation matching as any other label.
//!
//! DBSCAN is non-transductive: it implements `Fit` + `fit_predict` only, NO
//! standalone `predict` (D-08).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu
//! runs f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::dbscan::DBSCAN;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase};

/// DBSCAN fixture geometry (gen_oracle.py DB_N_SAMPLES × DB_N_FEATURES).
const DB_N_SAMPLES: usize = 40;
const DB_N_FEATURES: usize = 2;

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
        _ => unreachable!("dbscan fixtures are f32/f64 only"),
    }
}

/// Fit `DBSCAN::new(eps, min_samples)` on the fixture and return the host
/// `(labels_ as i64, core_sample_indices_ as i32)`.
fn fit_dbscan<F>(case: &OracleCase) -> (Vec<i64>, Vec<i32>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let eps = case.expect_f64("eps")[0];
    let min_samples = case.expect_f64("min_samples")[0] as usize;

    let db = DBSCAN::<F>::builder()
        .eps(eps)
        .min_samples(min_samples)
        .build::<F>()
        .expect("DBSCAN build with valid hyperparameters");
    let db = db
        .fit(&mut pool, &x_dev, None, (DB_N_SAMPLES, DB_N_FEATURES))
        .expect("DBSCAN::fit on a valid shape");

    let labels: Vec<i64> = db.labels(&pool).iter().map(|&l| l as i64).collect();
    let core: Vec<i32> = db.core_sample_indices(&pool);
    (labels, core)
}

/// `labels_` (noise = `-1`) match sklearn up to a cluster-label permutation
/// (D-09) AND `core_sample_indices_` match the fixture set EXACTLY (integer set,
/// no tolerance — it is a count threshold).
fn run_dbscan<F>(case: &OracleCase, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let labels_ref: Vec<i64> = case.expect_f64("labels").iter().map(|&v| v as i64).collect();
    let core_ref: Vec<i32> = case
        .expect_f64("core_sample_indices")
        .iter()
        .map(|&v| v as i32)
        .collect();
    assert_eq!(labels_ref.len(), DB_N_SAMPLES, "fixture labels len");

    let (labels, core) = fit_dbscan::<F>(case);

    // core_sample_indices_ EXACTLY matches the sklearn set (both ascending).
    let mut core_sorted = core.clone();
    core_sorted.sort_unstable();
    let mut core_ref_sorted = core_ref.clone();
    core_ref_sorted.sort_unstable();
    assert_eq!(
        core_sorted, core_ref_sorted,
        "{label}: core_sample_indices_ must EXACTLY match sklearn (integer set)"
    );

    // labels_ match sklearn up to a cluster-label permutation (the -1 noise label
    // is carried through the permutation matching like any other label, D-09).
    let acc = best_match_accuracy(&labels, &labels_ref);
    assert!(
        (acc - 1.0).abs() < f64::EPSILON,
        "{label}: best_match_accuracy {acc} != 1.0 (labels not a permutation of sklearn, noise=-1)"
    );
}

/// `labels_` (noise=-1) + `core_sample_indices_` vs sklearn up to a permutation,
/// f32.
#[test]
fn dbscan_labels_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("dbscan_f32_seed42.npz")).expect("load dbscan_f32");
    run_dbscan::<f32>(&case, "dbscan f32");
}

/// `core_sample_indices_` + labels vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn dbscan_core_sample_indices_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("dbscan f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    run_dbscan::<f64>(&case, "dbscan f64");
}

/// `fit_predict` returns the same labels as `fit` + `labels_` (noise=-1), and a
/// noise point IS present (the fixture is designed with ≥1 `-1`), exercising the
/// non-transductive `fit_predict` path (D-08 — there is no standalone predict).
#[test]
fn dbscan_fit_predict_consistency_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("dbscan_f32_seed42.npz")).expect("load dbscan_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let eps = case.expect_f64("eps")[0];
    let min_samples = case.expect_f64("min_samples")[0] as usize;

    let db = DBSCAN::<f32>::builder()
        .eps(eps)
        .min_samples(min_samples)
        .build::<f32>()
        .expect("DBSCAN build with valid hyperparameters");
    let (db, fp) = db
        .fit_predict(&mut pool, &x_dev, (DB_N_SAMPLES, DB_N_FEATURES))
        .expect("DBSCAN::fit_predict");
    let fp_labels: Vec<i32> = fp.to_host(&pool);
    let fitted_labels: Vec<i32> = db.labels(&pool);

    assert_eq!(
        fp_labels, fitted_labels,
        "fit_predict labels must equal the fitted labels_"
    );
    assert!(
        fp_labels.iter().any(|&l| l == -1),
        "DBSCAN fixture is designed with >=1 noise point (label -1)"
    );
}

/// Invalid hyperparameters are rejected at `build()` BEFORE any data is seen
/// (ASVS V5, the D-08 split): `eps < 0` → `BuildError::InvalidEps`,
/// `min_samples == 0` → `BuildError::InvalidMinSamples`. The data-INDEPENDENT
/// validation now lives in the builder, not the fit body (Phase 16 retrofit).
#[test]
fn dbscan_rejects_invalid_hyperparameters_f32() {
    // build() returns Result<DBSCAN<F, Unfit>, BuildError>; DBSCAN is not Debug,
    // so map the Ok arm away (.err()) before inspecting the error.
    let bad_eps = DBSCAN::<f32>::builder()
        .eps(-1.0)
        .min_samples(4)
        .build::<f32>()
        .err();
    assert!(
        matches!(
            bad_eps,
            Some(mlrs_algos::error::BuildError::InvalidEps { .. })
        ),
        "eps < 0 should surface BuildError::InvalidEps, got {bad_eps:?}"
    );

    let bad_min = DBSCAN::<f32>::builder()
        .eps(0.7)
        .min_samples(0)
        .build::<f32>()
        .err();
    assert!(
        matches!(
            bad_min,
            Some(mlrs_algos::error::BuildError::InvalidMinSamples { .. })
        ),
        "min_samples == 0 should surface BuildError::InvalidMinSamples, got {bad_min:?}"
    );
}

/// BLDR-01: `DBSCAN::new()` (the single-source defaults) equals
/// `DBSCAN::builder().build()` (the builder defaults re-derived from `new`).
#[test]
fn dbscan_defaults_equal() {
    let from_new = DBSCAN::<f32>::new();
    let from_builder = DBSCAN::<f32>::builder()
        .build::<f32>()
        .expect("default DBSCAN builder build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "DBSCAN::new() must equal DBSCAN::builder().build() (BLDR-01)"
    );
}
