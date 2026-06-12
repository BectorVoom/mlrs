//! Plan 05-02 — top-k select primitive (D-02) standalone oracle.
//!
//! Exercises the NEW `mlrs_backend::prims::topk::top_k` partial-select wrapper
//! over the Phase-2 pairwise-distance prim: per query row of `Xq`, select the `k`
//! nearest train points in `X` and assert the returned sqrt-Euclidean distances
//! match sklearn `NearestNeighbors.kneighbors` within 1e-5 AND the returned
//! neighbor indices match the fixture EXACTLY (lowest-index tie-break, D-02).
//! Runs STANDALONE before any KNN estimator (plan 08) consumes it — the D-01
//! primitive-first discipline.
//!
//! A `topk_lowest_index_tie_break` case feeds a CONSTRUCTED distance row with a
//! deliberate exact tie and asserts the LOWER column index is returned (mirrors
//! `reduce.rs::argmin_shared`, T-05-02-02 / Pitfall 8).
//!
//! ONE non-ignored test — `i32_device_array_roundtrips` — confirms D-06: an
//! `i32` `DeviceArray` (including the DBSCAN noise sentinel `-1`) round-trips
//! through the byte-keyed pool with ZERO pool/bridge changes.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, PrimError};

/// KNN fixture geometry (gen_oracle.py KNN_N_TRAIN/QUERY/FEATURES × KNN_K): the
/// per-query top-k distances/indices the top-k prim must reproduce.
const KNN_N_TRAIN: usize = 30;
const KNN_N_QUERY: usize = 8;
const KNN_N_FEATURES: usize = 3;
const KNN_K: usize = 5;

/// Distances vs sklearn `kneighbors` (sqrt-Euclidean) — the project 1e-5
/// contract. The fixture is well-spread (distinct distances, Pitfall 8) so the
/// f32 device path reaches well within 1e-5.
const DIST_TOL: f64 = 1e-5;

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("topk tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("topk tests are f32/f64 only"),
    }
}

/// `fixture-dtype` host vector from the f64 fixture array.
fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Shared oracle body: compose `distance(Xq, X, sqrt=false)` then
/// `top_k(.., k, sqrt=true)`, and assert the returned k distances (within 1e-5)
/// AND indices (exactly) match the sklearn `kneighbors` fixture for every query
/// row.
fn check_topk<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X"); // train, KNN_N_TRAIN × KNN_N_FEATURES
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq"); // query, KNN_N_QUERY × KNN_N_FEATURES
    let ref_dist: Vec<f64> = case.expect_f64("distances").to_vec(); // KNN_N_QUERY × KNN_K
    let ref_idx: Vec<f64> = case.expect_f64("indices").to_vec(); // KNN_N_QUERY × KNN_K (int-valued)

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    // Pairwise SQUARED Euclidean distance Xq×X = KNN_N_QUERY × KNN_N_TRAIN (no
    // sqrt — top-k selects on the squared form, Pitfall 8).
    let d_dev = distance::<F>(
        &mut pool,
        &xq_dev,
        (KNN_N_QUERY, KNN_N_FEATURES),
        &x_dev,
        (KNN_N_TRAIN, KNN_N_FEATURES),
        false,
        None,
    )
    .expect("pairwise distance on valid geometry");

    // Select the k nearest per query row, sqrt at the boundary so we get true
    // Euclidean distances to compare against sklearn.
    let (val_dev, idx_dev) = top_k::<F>(
        &mut pool,
        &d_dev,
        KNN_N_QUERY,
        KNN_N_TRAIN,
        KNN_K,
        true,
        None,
        None,
    )
    .expect("top_k on valid geometry");

    let got_dist: Vec<f64> = val_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let got_idx: Vec<u32> = idx_dev.to_host(&pool);
    val_dev.release_into(&mut pool);
    idx_dev.release_into(&mut pool);
    d_dev.release_into(&mut pool);

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
            let exp_idx = ref_idx[slot].round() as u32;
            assert_eq!(
                got_idx[slot], exp_idx,
                "index[{q}][{j}] mismatch vs sklearn: got={} expected={} \
                 (lowest-index tie-break, D-02)",
                got_idx[slot], exp_idx
            );
        }
    }
}

/// Resolve a workspace-root-relative fixture path (matches `cholesky_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the named array exists with exactly `len` flat elements.
fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

/// LOAD-NOT-JUST-PRESENT check: the committed `knn` fixture loads and its top-k
/// distance/index arrays are well-formed.
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

/// k smallest distances + exact indices per query row vs the sklearn
/// `kneighbors` reference, f32 (runs on cpu AND rocm).
#[test]
fn topk_distances_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_topk::<f32>("knn_f32_seed42.npz");
}

/// k smallest distances + exact indices per query row vs sklearn, f64 (cpu runs;
/// rocm skips-with-log).
#[test]
fn topk_distances_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("topk f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_topk::<f64>("knn_f64_seed42.npz");
}

/// Lowest-index tie-break on a CONSTRUCTED distance row with a deliberate exact
/// tie: two train points at the SAME distance must resolve to the LOWER column
/// index first (mirrors `reduce.rs::argmin_shared`, T-05-02-02 / Pitfall 8). f32,
/// runs on cpu AND rocm.
#[test]
fn topk_lowest_index_tie_break() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    println!("topk tie-break backend={backend} (T-05-02-02)");

    // One query row of 5 candidates: indices 1 and 3 share the distance 2.0; the
    // global minimum is index 2 (=0.0). The k=3 nearest must be, in order,
    // [2 (0.0), 1 (2.0), 3 (2.0)] — the tie between 1 and 3 broken by the LOWER
    // index (1 before 3). A naive last-writer-wins select would emit 3 before 1.
    let rows = 1usize;
    let cols = 5usize;
    let k = 3usize;
    let dist: Vec<f32> = vec![5.0, 2.0, 0.0, 2.0, 9.0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let d_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dist);

    // sqrt=false: compare on the raw distances as supplied (no GEMM-expansion
    // here — this is a direct, hand-built distance row).
    let (val_dev, idx_dev) =
        top_k::<f32>(&mut pool, &d_dev, rows, cols, k, false, None, None).expect("top_k tie row");
    let got_val = val_dev.to_host(&pool);
    let got_idx = idx_dev.to_host(&pool);
    val_dev.release_into(&mut pool);
    idx_dev.release_into(&mut pool);
    d_dev.release_into(&mut pool);

    assert_eq!(
        got_idx,
        vec![2u32, 1u32, 3u32],
        "tie-break: equal distances (idx 1 & 3 = 2.0) must resolve LOWER index \
         first; got idx {got_idx:?}"
    );
    assert_eq!(
        got_val,
        vec![0.0f32, 2.0f32, 2.0f32],
        "tie-break: values ascending [0.0, 2.0, 2.0]; got {got_val:?}"
    );
}

/// Geometry guard: a `k` outside `1..=cols` (and a `rows*cols != len`) must be
/// rejected with `PrimError::ShapeMismatch` BEFORE any launch (T-05-02-01 / ASVS
/// V5), never an out-of-bounds device read. f32, runs on cpu AND rocm.
#[test]
fn topk_rejects_bad_geometry() {
    let _ = env_logger::builder().is_test(true).try_init();
    let rows = 2usize;
    let cols = 4usize;
    let dist = vec![1.0f32; rows * cols];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let d_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dist);

    // k = 0 (< 1) is rejected.
    match top_k::<f32>(&mut pool, &d_dev, rows, cols, 0, false, None, None) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "k"),
        Err(other) => panic!("k=0 must be a ShapeMismatch on 'k', got {other:?}"),
        Ok(_) => panic!("k=0 must be rejected before launch, got Ok"),
    }
    // k > cols is rejected.
    match top_k::<f32>(&mut pool, &d_dev, rows, cols, cols + 1, false, None, None) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "k"),
        Err(other) => panic!("k>cols must be a ShapeMismatch on 'k', got {other:?}"),
        Ok(_) => panic!("k>cols must be rejected before launch, got Ok"),
    }
    // rows*cols != dist.len() is rejected.
    match top_k::<f32>(&mut pool, &d_dev, rows + 1, cols, 2, false, None, None) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "dist"),
        Err(other) => panic!("bad geometry must be a ShapeMismatch on 'dist', got {other:?}"),
        Ok(_) => panic!("bad geometry must be rejected before launch, got Ok"),
    }

    d_dev.release_into(&mut pool);
}

/// D-06 CONFIRMATION (non-ignored, runs on cpu AND rocm): an `i32` `DeviceArray`
/// round-trips through the byte-keyed pool with ZERO pool/bridge changes,
/// including the DBSCAN noise sentinel `-1`. This is the load-bearing D-06 check
/// the Wave-0 scaffold owes plans 05-04/05-10 (DBSCAN labels, KNN indices).
#[test]
fn i32_device_array_roundtrips() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    println!("i32 DeviceArray round-trip backend={backend} (D-06 confirmation)");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // -1 is the DBSCAN noise label; 0/5/42 are ordinary cluster/index ids.
    let host: [i32; 4] = [-1, 0, 5, 42];
    let dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(&mut pool, &host);
    let got: Vec<i32> = dev.to_host(&pool);
    dev.release_into(&mut pool);

    assert_eq!(
        got.as_slice(),
        host.as_slice(),
        "i32 DeviceArray must round-trip exactly (incl. the -1 DBSCAN noise value) \
         — confirms D-06 needs no pool/bridge changes"
    );
}
