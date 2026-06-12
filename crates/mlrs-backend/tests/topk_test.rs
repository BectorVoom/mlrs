//! Plan 05-02 — top-k select primitive (D-02) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub (04-01/03-02 precedent): every test function below that
//! references the not-yet-existing `prims::topk` symbol is `#[ignore]`d and
//! asserts ONLY that the committed `knn_{f32,f64}_seed42.npz` fixture loads and
//! its `distances`/`indices` arrays are well-formed (shape only) — so this test
//! crate COMPILES today against the empty `prims::topk` stub. Plan 05-02 removes
//! `#[ignore]` and wires the real top-k assertion (k smallest per query row,
//! lowest-index tie-break, sqrt-Euclidean distances within 1e-5).
//!
//! ONE non-ignored test — `i32_device_array_roundtrips` — confirms D-06: an
//! `i32` `DeviceArray` (including the DBSCAN noise sentinel `-1`) round-trips
//! through the byte-keyed pool with ZERO pool/bridge changes. This is the
//! load-bearing D-06 confirmation the plan owes downstream.
//!
//! f64 stubs carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// KNN fixture geometry (gen_oracle.py KNN_N_QUERY × KNN_K): the per-query top-k
/// distances/indices the top-k prim must reproduce.
const KNN_N_QUERY: usize = 8;
const KNN_K: usize = 5;

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
/// distance/index arrays are well-formed. WAVE-0 STUB — 05-02 removes `#[ignore]`
/// and wires the real top-k oracle on `prims::topk`.
#[test]
#[ignore = "Wave-0 scaffold: prims::topk not implemented until plan 05-02"]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "distances", KNN_N_QUERY * KNN_K);
    assert_len(&case, "indices", KNN_N_QUERY * KNN_K);
    assert_eq!(
        case.shape("distances"),
        Some([KNN_N_QUERY as u64, KNN_K as u64].as_slice())
    );
}

/// k smallest distances per query row vs the sklearn `kneighbors` reference, f32.
/// WAVE-0 STUB — 05-02 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::topk not implemented until plan 05-02"]
fn topk_distances_match_sklearn_f32() {
    let case = load_npz(fixture("knn_f32_seed42.npz")).expect("load knn_f32");
    assert_len(&case, "distances", KNN_N_QUERY * KNN_K);
}

/// k smallest distances per query row vs sklearn, f64 (cpu runs; rocm skips).
/// WAVE-0 STUB — 05-02 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::topk not implemented until plan 05-02"]
fn topk_distances_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "indices", KNN_N_QUERY * KNN_K);
}

/// Lowest-index tie-break on a constructed-tie distance row (D-02 convention).
/// WAVE-0 STUB — 05-02 wires the constructed-tie invariant on `prims::topk`.
#[test]
#[ignore = "Wave-0 scaffold: prims::topk not implemented until plan 05-02"]
fn topk_lowest_index_tie_break() {
    // 05-02 builds a synthetic distance row with a deliberate tie and asserts the
    // LOWER index wins (mirrors reduce.rs argmin_shared). Stub asserts fixture
    // load only for now.
    let case = load_npz(fixture("knn_f32_seed42.npz")).expect("load knn_f32");
    assert_len(&case, "k", 1);
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
