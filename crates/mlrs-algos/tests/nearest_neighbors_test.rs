//! Plan 05-08 — NearestNeighbors (NEIGH-01) sklearn oracle tests.
//!
//! Activated from the 05-01 Nyquist `#[ignore]` scaffold: each function loads the
//! committed `knn_{f32,f64}_seed42.npz` fixture, fits `NearestNeighbors` on the
//! train matrix `X`, calls `kneighbors(Xq, k)`, and asserts the returned
//! sqrt-Euclidean distances within 1e-5 of the fixture `distances` AND the
//! neighbor indices EXACTLY equal the fixture `indices` (lowest-index tie-break,
//! inherited from the validated top_k primitive, 05-02 / Pitfall 8).
//!
//! A `nearest_neighbors_rejects_bad_k` case pins the validate-before-launch
//! guard: `k` outside `1..=n_train` is rejected with `AlgoError::InvalidK`
//! (T-05-08-01 / ASVS V5).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::AlgoError;
use mlrs_algos::neighbors::nearest::NearestNeighbors;
use mlrs_algos::typestate::{Fit, KNeighbors};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// KNN fixture geometry (gen_oracle.py KNN_N_TRAIN/QUERY/FEATURES × KNN_K).
const KNN_N_TRAIN: usize = 30;
const KNN_N_QUERY: usize = 8;
const KNN_N_FEATURES: usize = 3;
const KNN_K: usize = 5;

/// Distances vs sklearn `kneighbors` (sqrt-Euclidean) — the project 1e-5
/// contract.
const DIST_TOL: f64 = 1e-5;

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

/// Shared oracle body: fit `NearestNeighbors` on `X`, call `kneighbors(Xq, k)`,
/// and assert the k distances (within 1e-5) AND indices (EXACTLY) match the
/// sklearn `kneighbors` fixture for every query row.
fn check_nearest_neighbors<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X"); // train KNN_N_TRAIN × KNN_N_FEATURES
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq"); // query KNN_N_QUERY × KNN_N_FEATURES
    let ref_dist: Vec<f64> = case.expect_f64("distances").to_vec();
    let ref_idx: Vec<f64> = case.expect_f64("indices").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<F>::builder()
        .n_neighbors(KNN_K)
        .build::<F>()
        .expect("build NearestNeighbors")
        .fit(&mut pool, &x_dev, None, (KNN_N_TRAIN, KNN_N_FEATURES))
        .expect("fit on valid geometry");

    let (val_dev, idx_dev) = nn
        .kneighbors(&mut pool, &xq_dev, (KNN_N_QUERY, KNN_N_FEATURES), KNN_K)
        .expect("kneighbors on valid geometry");

    let got_dist: Vec<f64> = val_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let got_idx: Vec<i32> = idx_dev.to_host(&pool);
    val_dev.release_into(&mut pool);
    idx_dev.release_into(&mut pool);

    // Distances within 1e-5, indices EXACT — for every (query, neighbor) slot.
    for q in 0..KNN_N_QUERY {
        for j in 0..KNN_K {
            let slot = q * KNN_K + j;
            let abs_err = (got_dist[slot] - ref_dist[slot]).abs();
            assert!(
                abs_err <= DIST_TOL + DIST_TOL * ref_dist[slot].abs(),
                "distance[{q}][{j}] mismatch vs sklearn: got={:e} expected={:e} abs_err={abs_err:e}",
                got_dist[slot],
                ref_dist[slot]
            );
            let exp_idx = ref_idx[slot].round() as i32;
            assert_eq!(
                got_idx[slot], exp_idx,
                "index[{q}][{j}] mismatch vs sklearn: got={} expected={} \
                 (lowest-index tie-break, NEIGH-01)",
                got_idx[slot], exp_idx
            );
        }
    }
}

/// LOAD-NOT-JUST-PRESENT: the `knn` fixture loads with well-formed
/// distances/indices (n_query × k).
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "distances", KNN_N_QUERY * KNN_K);
    assert_len(&case, "indices", KNN_N_QUERY * KNN_K);
    assert_eq!(
        case.shape("distances"),
        Some([KNN_N_QUERY as u64, KNN_K as u64].as_slice())
    );
}

/// BLDR-01: `NearestNeighbors::new()` equals `NearestNeighbors::builder().build()?`
/// on the hyperparameter subset (sklearn default `n_neighbors = 5`). Pure host
/// comparison — no device, so no f64 gate.
#[test]
fn defaults_equal() {
    let from_new = NearestNeighbors::<f64>::new();
    let from_builder = NearestNeighbors::<f64>::builder()
        .build::<f64>()
        .expect("default NearestNeighborsBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "NearestNeighbors::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// kneighbors distances + exact indices match sklearn, f32 (runs on cpu AND rocm).
#[test]
fn nearest_neighbors_distances_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_nearest_neighbors::<f32>("knn_f32_seed42.npz");
}

/// kneighbors distances + exact indices match sklearn, f64 (cpu runs;
/// rocm skips-with-log).
#[test]
fn nearest_neighbors_indices_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("nearest_neighbors f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    check_nearest_neighbors::<f64>("knn_f64_seed42.npz");
}

/// Validate-before-launch guard (T-05-08-01 / ASVS V5): a `k` outside
/// `1..=n_train` is rejected with `AlgoError::InvalidK` BEFORE any prim launch.
/// f32, runs on cpu AND rocm.
#[test]
fn nearest_neighbors_rejects_bad_k() {
    let _ = env_logger::builder().is_test(true).try_init();
    let case = load_npz(fixture("knn_f32_seed42.npz")).expect("load knn_f32");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let xq: Vec<f32> = fixture_vec::<f32>(&case, "Xq");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<f32>::builder()
        .n_neighbors(KNN_K)
        .build::<f32>()
        .expect("build NearestNeighbors")
        .fit(&mut pool, &x_dev, None, (KNN_N_TRAIN, KNN_N_FEATURES))
        .expect("fit on valid geometry");

    // k = 0 (< 1) is rejected.
    match nn.kneighbors(&mut pool, &xq_dev, (KNN_N_QUERY, KNN_N_FEATURES), 0) {
        Err(AlgoError::InvalidK { k, n_samples, .. }) => {
            assert_eq!(k, 0);
            assert_eq!(n_samples, KNN_N_TRAIN);
        }
        Err(other) => panic!("k=0 must be AlgoError::InvalidK, got {other:?}"),
        Ok(_) => panic!("k=0 must be rejected before launch, got Ok"),
    }
    // k > n_train is rejected.
    match nn.kneighbors(
        &mut pool,
        &xq_dev,
        (KNN_N_QUERY, KNN_N_FEATURES),
        KNN_N_TRAIN + 1,
    ) {
        Err(AlgoError::InvalidK { k, n_samples, .. }) => {
            assert_eq!(k, KNN_N_TRAIN + 1);
            assert_eq!(n_samples, KNN_N_TRAIN);
        }
        Err(other) => panic!("k>n_train must be AlgoError::InvalidK, got {other:?}"),
        Ok(_) => panic!("k>n_train must be rejected before launch, got Ok"),
    }
}
