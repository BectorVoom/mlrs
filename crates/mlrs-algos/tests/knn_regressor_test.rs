//! Plan 05-08 — KNeighborsRegressor (NEIGH-03) sklearn oracle tests.
//!
//! Activated from the 05-01 Nyquist `#[ignore]` scaffold: each function loads the
//! committed `knn_{f32,f64}_seed42.npz` fixture, fits `KNeighborsRegressor` on
//! `(X, y_reg)`, and asserts `predict` (mean of the k neighbor targets, uniform
//! weights, via the `Predict<F>` surface) matches the fixture `predict_reg`
//! within 1e-5.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). f32 runs on rocm. Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
use mlrs_algos::typestate::{Fit, Predict};
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

const PRED_TOL: f64 = 1e-5;

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

/// Shared oracle body: fit on `(X, y_reg)`, assert `predict` (neighbor mean)
/// within 1e-5 of the fixture `predict_reg` for every query.
fn check_regressor<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let y_reg: Vec<F> = fixture_vec::<F>(&case, "y_reg");
    let ref_predict: Vec<f64> = case.expect_f64("predict_reg").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_reg);

    let reg = KNeighborsRegressor::<F>::builder()
        .n_neighbors(KNN_K)
        .build::<F>()
        .expect("build KNeighborsRegressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (KNN_N_TRAIN, KNN_N_FEATURES))
        .expect("fit on valid geometry");

    let pred_dev = reg
        .predict(&mut pool, &xq_dev, (KNN_N_QUERY, KNN_N_FEATURES))
        .expect("predict on valid geometry");
    let got: Vec<f64> = pred_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    pred_dev.release_into(&mut pool);

    for q in 0..KNN_N_QUERY {
        let abs_err = (got[q] - ref_predict[q]).abs();
        assert!(
            abs_err <= PRED_TOL + PRED_TOL * ref_predict[q].abs(),
            "predict[{q}] mismatch vs sklearn: got={:e} expected={:e} abs_err={abs_err:e} \
             (neighbor mean, NEIGH-03)",
            got[q],
            ref_predict[q]
        );
    }
}

/// LOAD-NOT-JUST-PRESENT: the `knn` fixture loads with well-formed y_reg
/// (n_train) + predict_reg (n_query).
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "y_reg", KNN_N_TRAIN);
    assert_len(&case, "predict_reg", KNN_N_QUERY);
}

/// BLDR-01: `KNeighborsRegressor::new()` equals
/// `KNeighborsRegressor::builder().build()?` on the hyperparameter subset
/// (sklearn default `n_neighbors = 5`). Pure host comparison — no device.
#[test]
fn defaults_equal() {
    let from_new = KNeighborsRegressor::<f64>::new();
    let from_builder = KNeighborsRegressor::<f64>::builder()
        .build::<f64>()
        .expect("default KNeighborsRegressorBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "KNeighborsRegressor::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// predict (neighbor mean) matches sklearn within 1e-5, f32 (cpu AND rocm).
#[test]
fn knn_regressor_predict_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_regressor::<f32>("knn_f32_seed42.npz");
}

/// predict matches sklearn within 1e-5, f64 (cpu runs; rocm skips-with-log).
#[test]
fn knn_regressor_predict_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("knn_regressor f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    check_regressor::<f64>("knn_f64_seed42.npz");
}
