//! Plan 05-08 — KNeighborsClassifier (NEIGH-02) sklearn oracle tests.
//!
//! Activated from the 05-01 Nyquist `#[ignore]` scaffold: each function loads the
//! committed `knn_{f32,f64}_seed42.npz` fixture, fits `KNeighborsClassifier` on
//! `(X, y_class)`, and asserts `predict_labels` (majority vote, i32) matches the
//! fixture `predict_class` EXACTLY AND `predict_proba` (per-class neighbor
//! fraction, uniform weights) matches the fixture `predict_proba` within 1e-5.
//! The argmax vote uses the lowest-class-index tie-break (`argmax_rows`, 02).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). f32 runs on rocm. Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::classifier::KNeighborsClassifier;
use mlrs_algos::traits::{Fit, PredictLabels, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// KNN fixture geometry (gen_oracle.py).
const KNN_N_TRAIN: usize = 30;
const KNN_N_QUERY: usize = 8;
const KNN_N_FEATURES: usize = 3;
const KNN_K: usize = 5;
const KNN_N_CLASSES: usize = 3;

const PROBA_TOL: f64 = 1e-5;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn fixtures are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("knn fixtures are f32/f64 only"),
    }
}

fn fixture_vec<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

/// Shared oracle body: fit on `(X, y_class)`, assert `predict_labels` matches
/// `predict_class` EXACTLY and `predict_proba` within 1e-5 of `predict_proba`.
fn check_classifier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let y_class: Vec<F> = fixture_vec::<F>(&case, "y_class");
    let ref_predict: Vec<f64> = case.expect_f64("predict_class").to_vec();
    let ref_proba: Vec<f64> = case.expect_f64("predict_proba").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_class);

    let mut clf = KNeighborsClassifier::<F>::new(KNN_K);
    clf.fit(&mut pool, &x_dev, Some(&y_dev), (KNN_N_TRAIN, KNN_N_FEATURES))
        .expect("fit on valid geometry");
    assert_eq!(
        clf.n_classes().expect("fitted"),
        KNN_N_CLASSES,
        "inferred n_classes (max+1) should match the fixture"
    );

    // predict_proba within 1e-5 for every (query, class) slot.
    let proba_dev = clf
        .predict_proba(&mut pool, &xq_dev, (KNN_N_QUERY, KNN_N_FEATURES))
        .expect("predict_proba on valid geometry");
    let got_proba: Vec<f64> = proba_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    proba_dev.release_into(&mut pool);
    for q in 0..KNN_N_QUERY {
        for c in 0..KNN_N_CLASSES {
            let slot = q * KNN_N_CLASSES + c;
            let abs_err = (got_proba[slot] - ref_proba[slot]).abs();
            assert!(
                abs_err <= PROBA_TOL + PROBA_TOL * ref_proba[slot].abs(),
                "proba[{q}][{c}] mismatch vs sklearn: got={:e} expected={:e} abs_err={abs_err:e}",
                got_proba[slot],
                ref_proba[slot]
            );
        }
    }

    // predict (majority vote, i32) matches the fixture EXACTLY (lowest-class-index
    // tie-break).
    let labels_dev = clf
        .predict_labels(&mut pool, &xq_dev, (KNN_N_QUERY, KNN_N_FEATURES))
        .expect("predict_labels on valid geometry");
    let got_labels: Vec<i32> = labels_dev.to_host(&pool);
    labels_dev.release_into(&mut pool);
    for q in 0..KNN_N_QUERY {
        let exp = ref_predict[q].round() as i32;
        assert_eq!(
            got_labels[q], exp,
            "predict[{q}] mismatch vs sklearn: got={} expected={} \
             (majority vote, lowest-class-index tie, NEIGH-02)",
            got_labels[q], exp
        );
    }
}

/// LOAD-NOT-JUST-PRESENT: the `knn` fixture loads with well-formed
/// predict_class (n_query) + predict_proba (n_query × n_classes).
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "predict_class", KNN_N_QUERY);
    assert_len(&case, "predict_proba", KNN_N_QUERY * KNN_N_CLASSES);
    assert_len(&case, "y_class", KNN_N_TRAIN);
}

/// predict (majority vote, i32) + predict_proba match sklearn, f32 (cpu AND rocm).
#[test]
fn knn_classifier_predict_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_classifier::<f32>("knn_f32_seed42.npz");
}

/// predict + predict_proba match sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn knn_classifier_predict_proba_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("knn_classifier f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    check_classifier::<f64>("knn_f64_seed42.npz");
}
