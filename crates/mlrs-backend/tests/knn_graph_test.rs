//! Plan 13-01 — multi-metric KNN-graph primitive (PRIM-11) Nyquist Wave-0
//! oracle harness. **RED-BY-DESIGN**: this file references
//! `mlrs_backend::prims::knn_graph::{knn_graph, Metric}`, which plan 13-03 lands.
//! Until then `cargo test -p mlrs-backend --features cpu --test knn_graph_test`
//! fails with an UNRESOLVED-SYMBOL error (not a stubbed pass) — confirming the
//! gate is real before plans 02/03 land RED→GREEN against it.
//!
//! The harness exercises the shared directed KNN-graph prim over the full fixed
//! metric set (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p) against
//! per-metric `sklearn.neighbors.NearestNeighbors` oracles (plan 13-01 fixtures,
//! X-vs-X, `k+1` self-inclusive neighbours). It asserts:
//!
//!   1. per-row index SET-EQUALITY vs sklearn up to tie-ordering (NOT exact
//!      index — cross-metric tie-ordering differs; PRIM-11),
//!   2. distances within `DIST_TOL = 1e-5` relative tolerance,
//!   3. `knn_rejects_bad_geometry` — `k > n-1` (include_self=false), `p < 1`
//!      (Minkowski), and `n*d != len` are each `Err(ShapeMismatch{operand,..})`
//!      with operands `"k"` / `"p"` / `"x"` BEFORE launch,
//!   4. `knn_self_drop_duplicate_point_value` (R-9) — the LOAD-BEARING gate: on
//!      the duplicate-point fixture with `include_self=false`, neighbour 0 of a
//!      duplicate's query row is the GENUINE duplicate index (NOT self) AND the
//!      distance VALUES match the brute-force reference. This is the ONLY catch
//!      for the cpu-MLIR SILENT self-drop miscompile (FINDING 002-B) — it
//!      asserts VALUES, not non-panic.
//!   5. `knn_include_self_returns_self_at_col0` (HDBSCAN core-distance path) —
//!      with `include_self=true` self is present in every row at column 0,
//!   6. `knn_memory_gate_query_axis_tiled` — one threaded `BufferPool`, the prim
//!      kept SUB-QUADRATIC in `n` (never full `n×n` resident), `live_bytes`
//!      conserves after warmup, scratch `reuses > 0` (HARD build-failing).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log). Per AGENTS.md §2 tests live here, never an in-source
//! `#[cfg(test)] mod tests`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
// RED-BY-DESIGN (plan 13-03 lands these): the prim + its metric enum. The whole
// file fails to compile on an UNRESOLVED SYMBOL until plan 13-03 — a REAL gate,
// not a stubbed pass.
use mlrs_backend::prims::knn_graph::{knn_graph, Metric};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, PrimError};

/// KNN-graph fixture geometry (gen_oracle.py KNN_N_TRAIN/FEATURES × KNN_K). The
/// fixtures are X-vs-X (the graph queries the train set against itself) and store
/// the `k+1` self-inclusive sklearn neighbours.
const N: usize = 30; // KNN_N_TRAIN
const D: usize = 3; // KNN_N_FEATURES
const K: usize = 5; // KNN_K — true neighbours requested
const K1: usize = K + 1; // the stored self-inclusive neighbour count
/// Fixed non-degenerate Minkowski exponent (gen_oracle.py KNN_METRIC_P).
const MINKOWSKI_P: f64 = 3.0;

/// Distances vs sklearn `kneighbors` — the project 1e-5 contract (relative tol).
const DIST_TOL: f64 = 1e-5;

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("knn tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("knn tests are f32/f64 only"),
    }
}

/// `fixture-dtype` host vector from the f64 fixture array.
fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Resolve a workspace-root-relative fixture path (matches `topk_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// The Minkowski exponent the prim should be handed for a metric. Non-Minkowski
/// metrics carry an unused `2.0` (validated `>= 1`, ignored by the kernel route).
fn metric_p(metric: Metric) -> f64 {
    match metric {
        Metric::Minkowski { p } => p,
        _ => 2.0,
    }
}

/// Shared per-metric oracle body: run `knn_graph` over the X-vs-X fixture with
/// `include_self=false` (the directed UMAP path — `k` true neighbours, self
/// dropped) and assert (1) per-row index SET-EQUALITY vs the sklearn `k+1`
/// neighbour set MINUS self, up to tie-ordering, and (2) distances within
/// `DIST_TOL` relative tolerance. Indices are compared as SETS (per-row
/// `BTreeSet`), NOT exact position — cross-metric tie-ordering differs.
fn check_knn_metric<F>(fixture_name: &str, metric: Metric)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn metric fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X"); // N × D, queried against itself
    let ref_dist: Vec<f64> = case.expect_f64("distances").to_vec(); // N × K1
    let ref_idx: Vec<f64> = case.expect_f64("indices").to_vec(); // N × K1 (int-valued)

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);

    // Directed graph: k true neighbours per row, self dropped by INDEX IDENTITY.
    let (idx_dev, val_dev) = knn_graph::<F>(
        &mut pool,
        &x_dev,
        (N, D),
        K,
        metric,
        /* include_self */ false,
        metric_p(metric),
    )
    .expect("knn_graph on valid geometry");

    let got_idx: Vec<u32> = idx_dev.to_host(&pool);
    let got_dist: Vec<f64> = val_dev
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    idx_dev.release_into(&mut pool);
    val_dev.release_into(&mut pool);

    for row in 0..N {
        // sklearn neighbour set for this row, with the self column (index == row)
        // removed — that is the directed include_self=false oracle.
        let mut want_idx: BTreeSet<u32> = BTreeSet::new();
        let mut want_dist: Vec<f64> = Vec::with_capacity(K);
        for j in 0..K1 {
            let slot = row * K1 + j;
            let nb = ref_idx[slot].round() as u32;
            if nb as usize == row {
                continue; // self column — dropped by the prim
            }
            want_idx.insert(nb);
            want_dist.push(ref_dist[slot]);
        }
        // If self was NOT present in the top-(k+1) (should not happen X-vs-X),
        // the oracle still only holds K entries — trim the farthest.
        want_dist.truncate(K);

        let got_set: BTreeSet<u32> = (0..K).map(|j| got_idx[row * K + j]).collect();
        assert_eq!(
            got_set,
            want_idx,
            "metric {metric:?} row {row}: index SET mismatch vs sklearn (up to \
             tie-ordering). got={got_set:?} want={want_idx:?}"
        );

        // Distances are order-independent here (set compare), so compare the
        // SORTED distance vectors within the 1e-5 relative tol.
        let mut got_row: Vec<f64> = (0..K).map(|j| got_dist[row * K + j]).collect();
        got_row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want_row = want_dist.clone();
        want_row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for j in 0..K {
            let abs_err = (got_row[j] - want_row[j]).abs();
            assert!(
                abs_err <= DIST_TOL + DIST_TOL * want_row[j].abs(),
                "metric {metric:?} row {row} neighbour {j}: distance mismatch vs \
                 sklearn: got={:e} expected={:e} abs_err={abs_err:e}",
                got_row[j],
                want_row[j]
            );
        }
    }
}

// ===========================================================================
// Per-metric oracle tests — f64 (cpu gate; rocm skips-with-log) + f32 companion
// (cpu AND rocm). Index SET-equality + 1e-5 distances vs sklearn.
// ===========================================================================

macro_rules! metric_oracle_pair {
    ($f64_name:ident, $f32_name:ident, $fixture_stem:literal, $metric:expr) => {
        #[test]
        fn $f64_name() {
            let _ = env_logger::builder().is_test(true).try_init();
            let backend = capability::active_backend_name();
            capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
            if capability::skip_f64_with_log() {
                println!("knn f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
                return;
            }
            check_knn_metric::<f64>(concat!($fixture_stem, "_f64_seed42.npz"), $metric);
        }

        #[test]
        fn $f32_name() {
            let _ = env_logger::builder().is_test(true).try_init();
            let backend = capability::active_backend_name();
            capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
            check_knn_metric::<f32>(concat!($fixture_stem, "_f32_seed42.npz"), $metric);
        }
    };
}

metric_oracle_pair!(
    knn_euclidean_matches_sklearn_f64,
    knn_euclidean_matches_sklearn_f32,
    "knn_euclidean",
    Metric::Euclidean
);
metric_oracle_pair!(
    knn_manhattan_matches_sklearn_f64,
    knn_manhattan_matches_sklearn_f32,
    "knn_manhattan",
    Metric::Manhattan
);
metric_oracle_pair!(
    knn_cosine_matches_sklearn_f64,
    knn_cosine_matches_sklearn_f32,
    "knn_cosine",
    Metric::Cosine
);
metric_oracle_pair!(
    knn_chebyshev_matches_sklearn_f64,
    knn_chebyshev_matches_sklearn_f32,
    "knn_chebyshev",
    Metric::Chebyshev
);
metric_oracle_pair!(
    knn_minkowski_matches_sklearn_f64,
    knn_minkowski_matches_sklearn_f32,
    "knn_minkowski",
    Metric::Minkowski { p: MINKOWSKI_P }
);

// ===========================================================================
// Geometry-rejection gate (ASVS V5): bad geometry rejected BEFORE any launch.
// ===========================================================================

/// `k > n-1` (include_self=false → needs k+1 distinct rows), `p < 1` (Minkowski),
/// and `n*d != len` are each `Err(PrimError::ShapeMismatch { operand, .. })`
/// (operands `"k"` / `"p"` / `"x"`) BEFORE launch — never an OOB device read.
/// f32, runs on cpu AND rocm.
#[test]
fn knn_rejects_bad_geometry() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    // A tiny valid design: n=4 rows, d=2.
    let n = 4usize;
    let d = 2usize;
    let x: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.5).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    // k > n-1 with include_self=false: needs k+1 <= n, so k=n is rejected on "k".
    match knn_graph::<f32>(&mut pool, &x_dev, (n, d), n, Metric::Euclidean, false, 2.0) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "k"),
        Err(other) => panic!("k>n-1 must be a ShapeMismatch on 'k', got {other:?}"),
        Ok(_) => panic!("k>n-1 must be rejected before launch, got Ok"),
    }

    // p < 1 (Minkowski) is rejected on "p".
    match knn_graph::<f32>(
        &mut pool,
        &x_dev,
        (n, d),
        1,
        Metric::Minkowski { p: 0.5 },
        false,
        0.5,
    ) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "p"),
        Err(other) => panic!("p<1 must be a ShapeMismatch on 'p', got {other:?}"),
        Ok(_) => panic!("p<1 must be rejected before launch, got Ok"),
    }

    // n*d != x.len() is rejected on "x".
    match knn_graph::<f32>(&mut pool, &x_dev, (n + 1, d), 1, Metric::Euclidean, false, 2.0) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "x"),
        Err(other) => panic!("bad geometry must be a ShapeMismatch on 'x', got {other:?}"),
        Ok(_) => panic!("bad geometry must be rejected before launch, got Ok"),
    }

    x_dev.release_into(&mut pool);
}

// ===========================================================================
// R-9 — the LOAD-BEARING duplicate-point VALUE gate (the only catch for the
// cpu-MLIR SILENT self-drop miscompile, FINDING 002-B). Asserts VALUES.
// ===========================================================================

/// On the duplicate-point fixture (train rows `dup_row_a` / `dup_row_b` are
/// identical) with `include_self=false`, neighbour 0 of the duplicate's query
/// row MUST be the GENUINE duplicate index (NOT the self index) — proving
/// index-IDENTITY self-drop (D-02), not "first zero-distance" — AND the returned
/// distance VALUE must be ~0 (the duplicate sits at distance 0). A silent
/// miscompile (002-B) passes a happy-path non-panic check; only this VALUE
/// assertion on distance-0 duplicates catches it. f64 on the cpu gate.
#[test]
fn knn_self_drop_duplicate_point_value() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "dup-point R-9");
    if capability::skip_f64_with_log() {
        println!("knn dup-point f64 backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }

    let case = load_npz(fixture("knn_euclidean_f64_seed42.npz")).expect("load dup fixture");
    let x: Vec<f64> = case.expect_f64("X").to_vec();
    let dup_a = case.expect_f64("dup_row_a")[0].round() as usize;
    let dup_b = case.expect_f64("dup_row_b")[0].round() as usize;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let (idx_dev, val_dev) =
        knn_graph::<f64>(&mut pool, &x_dev, (N, D), K, Metric::Euclidean, false, 2.0)
            .expect("knn_graph dup-point");
    let got_idx: Vec<u32> = idx_dev.to_host(&pool);
    let got_dist: Vec<f64> = val_dev.to_host(&pool);
    idx_dev.release_into(&mut pool);
    val_dev.release_into(&mut pool);

    // Query row `dup_a`: self (index dup_a) MUST be dropped, and the GENUINE
    // duplicate (index dup_b, distance 0) must be neighbour 0 — by IDENTITY.
    let n0_idx = got_idx[dup_a * K];
    let n0_dist = got_dist[dup_a * K];
    assert_eq!(
        n0_idx as usize, dup_b,
        "R-9 (002-B catch): row {dup_a} neighbour 0 must be the GENUINE duplicate \
         index {dup_b} (self-drop by INDEX IDENTITY), not self {dup_a}; got {n0_idx}"
    );
    assert!(
        n0_dist.abs() <= DIST_TOL,
        "R-9: row {dup_a} neighbour-0 distance to its duplicate must be ~0, got {n0_dist:e}"
    );
    // Symmetric check on the partner row.
    let m0_idx = got_idx[dup_b * K];
    let m0_dist = got_dist[dup_b * K];
    assert_eq!(
        m0_idx as usize, dup_a,
        "R-9: row {dup_b} neighbour 0 must be the genuine duplicate {dup_a}, got {m0_idx}"
    );
    assert!(
        m0_dist.abs() <= DIST_TOL,
        "R-9: row {dup_b} neighbour-0 distance must be ~0, got {m0_dist:e}"
    );
    // Self must NOT appear anywhere in either row's directed neighbour list.
    for j in 0..K {
        assert_ne!(
            got_idx[dup_a * K + j] as usize, dup_a,
            "R-9: self index {dup_a} must NOT survive include_self=false self-drop"
        );
        assert_ne!(
            got_idx[dup_b * K + j] as usize, dup_b,
            "R-9: self index {dup_b} must NOT survive include_self=false self-drop"
        );
    }
}

// ===========================================================================
// HDBSCAN core-distance path — include_self=true puts self at column 0.
// ===========================================================================

/// With `include_self=true` (HDBSCAN core-distance path) every row's nearest
/// neighbour is ITSELF at column 0 (distance 0) — `top_k(k)` returns self
/// naturally, no self-drop. f64 on the cpu gate.
#[test]
fn knn_include_self_returns_self_at_col0() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "include_self");
    if capability::skip_f64_with_log() {
        println!("knn include_self f64 backend={backend}: SKIPPED (no f64 on this adapter)");
        return;
    }

    let case = load_npz(fixture("knn_euclidean_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = case.expect_f64("X").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let (idx_dev, val_dev) =
        knn_graph::<f64>(&mut pool, &x_dev, (N, D), K, Metric::Euclidean, true, 2.0)
            .expect("knn_graph include_self");
    let got_idx: Vec<u32> = idx_dev.to_host(&pool);
    let got_dist: Vec<f64> = val_dev.to_host(&pool);
    idx_dev.release_into(&mut pool);
    val_dev.release_into(&mut pool);

    for row in 0..N {
        // Skip the duplicate rows: with a genuine distance-0 duplicate present,
        // column 0 may be EITHER self or the duplicate (both at distance 0) — the
        // lowest-index tie-break decides. The non-duplicate rows pin self@col0.
        let dup_a = case.expect_f64("dup_row_a")[0].round() as usize;
        let dup_b = case.expect_f64("dup_row_b")[0].round() as usize;
        if row == dup_a || row == dup_b {
            // Either self or its duplicate is acceptable at col 0; distance is 0.
            assert!(
                got_dist[row * K].abs() <= DIST_TOL,
                "include_self: dup row {row} col-0 distance must be ~0, got {:e}",
                got_dist[row * K]
            );
            continue;
        }
        assert_eq!(
            got_idx[row * K] as usize, row,
            "include_self=true: row {row} neighbour 0 must be SELF, got {}",
            got_idx[row * K]
        );
        assert!(
            got_dist[row * K].abs() <= DIST_TOL,
            "include_self=true: row {row} self distance must be ~0, got {:e}",
            got_dist[row * K]
        );
    }
}

// ===========================================================================
// Memory gate (R-6) — query-axis-tiled, never full n×n resident-and-leaking.
// HARD, build-failing PoolStats assertions (drive f32, backend-agnostic).
// ===========================================================================

/// Thread ONE `BufferPool` and run `knn_graph` repeatedly at a fixture size,
/// asserting the query-axis-tiled composition (R-6): `peak_bytes` stays
/// SUB-QUADRATIC in `n` (the prim never materializes a full `n×n` distance block
/// resident), `live_bytes` CONSERVES after a warmup iteration (transient scratch
/// released, the CR-02 honesty signal), and scratch `reuses` GROW (the free-list
/// serves same-shape scratch). These are HARD `assert!`s — the suite goes red if
/// the device-residency / tiling contract breaks. Threshold tuning is deferred to
/// plan 13-03 once the tile size is finalized; the SUB-QUADRATIC and conservation
/// gates here are the structural invariants that must hold regardless.
#[test]
fn knn_memory_gate_query_axis_tiled() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    let case = load_npz(fixture("knn_euclidean_f32_seed42.npz")).expect("load fixture");
    let x: Vec<f32> = case.expect_f32("X").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    const ITERS: usize = 4;
    let mut live_after: Vec<u64> = Vec::with_capacity(ITERS);
    let mut peak_after: Vec<u64> = Vec::with_capacity(ITERS);
    let mut reuses_after: Vec<u64> = Vec::with_capacity(ITERS);

    for _iter in 0..ITERS {
        let (idx_dev, val_dev) =
            knn_graph::<f32>(&mut pool, &x_dev, (N, D), K, Metric::Euclidean, false, 2.0)
                .expect("knn_graph memory gate");
        idx_dev.release_into(&mut pool);
        val_dev.release_into(&mut pool);
        let s = pool.stats();
        live_after.push(s.live_bytes);
        peak_after.push(s.peak_bytes);
        reuses_after.push(s.reuses);
    }

    // SUB-QUADRATIC residency: peak_bytes must be far below a full n×n f32
    // distance matrix being resident-and-leaking across iterations. A query-axis
    // tile keeps only a (tile × n) block live; even an untiled single-block
    // build keeps ONE n×n transient that is RELEASED — so peak must not scale
    // with ITERS. Bound peak by a generous multiple of one n×n block: if the
    // prim leaked an n×n block per iteration, peak would exceed ITERS×n×n.
    let nn_block = (N * N * std::mem::size_of::<f32>()) as u64;
    assert!(
        peak_after[ITERS - 1] < (ITERS as u64) * nn_block,
        "R-6 (sub-quadratic residency) FAILED on {backend}: peak_bytes={} >= \
         ITERS({ITERS})×n×n({nn_block}) — the prim is leaking a full n×n block \
         per call (not query-axis tiled / not releasing scratch). stats={:?}",
        peak_after[ITERS - 1],
        pool.stats()
    );

    // live_bytes does NOT GROW after warmup (transient scratch released back to
    // the persistent footprint — the CR-02 honesty signal; growing live = leak).
    // WR-05: a non-growth BOUND, not exact byte equality — exact equality would
    // flip to a false red on any benign allocator-rounding / one-shot-cached-
    // scratch change that does not actually leak. A real leak still trips this
    // (live climbs each call).
    let live_baseline = live_after[1];
    for iter in 2..ITERS {
        assert!(
            live_after[iter] <= live_baseline,
            "R-6 (live_bytes conserved) FAILED on {backend}: iter {iter} \
             live_bytes={} > baseline={live_baseline} — transient KNN scratch \
             is NOT being released (it climbs each call → leak). stats={:?}",
            live_after[iter],
            pool.stats()
        );
    }

    // peak_bytes does NOT GROW after warmup (released scratch reused in place).
    // WR-05: a non-growth BOUND (see live_bytes rationale above) — a leak makes
    // peak climb with the call count, which this still catches.
    let peak_baseline = peak_after[1];
    for iter in 2..ITERS {
        assert!(
            peak_after[iter] <= peak_baseline,
            "R-6 (peak_bytes bounded) FAILED on {backend}: iter {iter} \
             peak_bytes={} > baseline={peak_baseline} — peak grows with the call \
             count (scratch not released → buffers stack). stats={:?}",
            peak_after[iter],
            pool.stats()
        );
    }

    // scratch reuse GROWS each steady-state iteration — the free-list serves the
    // same-shape KNN scratch (distance block, top_k(k+1) intermediate). Zero
    // would mean nothing is released-then-reacquired.
    let delta = reuses_after[ITERS - 1] - reuses_after[ITERS - 2];
    assert!(
        delta > 0,
        "R-6 (scratch reuse grows) FAILED on {backend}: steady-state reuse \
         delta={delta} — no per-call KNN scratch reuse. stats={:?}",
        pool.stats()
    );

    x_dev.release_into(&mut pool);
    println!(
        "R-6 knn memory gate backend={backend}: live_baseline={live_baseline} \
         peak_baseline={peak_baseline} reuse_delta={delta} final_stats={:?}",
        pool.stats()
    );
}
