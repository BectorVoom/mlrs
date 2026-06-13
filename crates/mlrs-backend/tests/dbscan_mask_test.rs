//! Plan 05-04 — DBSCAN eps-core-mask primitive (D-04) standalone oracle.
//!
//! Exercises the NEW `mlrs_backend::prims::dbscan::eps_core_mask` wrapper over the
//! Phase-2 pairwise-distance prim: the DEVICE builds the `n × n` squared-distance
//! matrix, thresholds it at `eps²`, and counts each point's self-inclusive
//! eps-neighbors; the host reads the count + adjacency back (the D-04 documented
//! round-trip) and derives `is_core[i] = count[i] >= min_samples`. The resulting
//! core mask must EXACTLY match sklearn `DBSCAN.core_sample_indices_` — integer
//! exact (it is a count threshold, no tolerance). Runs STANDALONE before the
//! DBSCAN estimator (plan 07) consumes it — the D-01 primitive-first discipline.
//!
//! A self-inclusivity case pins Pitfall 7: every point is its OWN eps-neighbor, so
//! `count[i] >= 1` for all `i` (`D[i,i] = 0 <= eps²`).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::dbscan::eps_core_mask;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

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

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("dbscan tests are f32/f64 only"),
    }
}

fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Shared oracle body: run `eps_core_mask(X, n, d, eps, min_samples)` and assert
/// the resulting `is_core` mask EXACTLY matches sklearn's `core_sample_indices_`
/// (`is_core[i]` true iff `i ∈ core_sample_indices`). Integer-exact — it is a
/// count threshold, not a float reduction (no tolerance).
fn check_dbscan_mask<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load dbscan fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X"); // DB_N_SAMPLES × DB_N_FEATURES
    let eps = case.expect_f64("eps")[0];
    let min_samples = case.expect_f64("min_samples")[0].round() as u32;
    // sklearn core_sample_indices_ (int-valued) — the reference core SET.
    let core_idx: Vec<usize> = case
        .expect_f64("core_sample_indices")
        .iter()
        .map(|&v| v.round() as usize)
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);

    let mask = eps_core_mask::<F>(
        &mut pool,
        &x_dev,
        DB_N_SAMPLES,
        DB_N_FEATURES,
        eps,
        min_samples,
    )
    .expect("eps_core_mask on valid geometry");
    x_dev.release_into(&mut pool);

    // Build the expected boolean mask from the sklearn core index SET.
    let mut expected = vec![false; DB_N_SAMPLES];
    for &i in &core_idx {
        assert!(
            i < DB_N_SAMPLES,
            "core_sample_index {i} out of bounds (n={DB_N_SAMPLES})"
        );
        expected[i] = true;
    }

    assert_eq!(
        mask.is_core.len(),
        DB_N_SAMPLES,
        "is_core mask length must be n_samples"
    );
    // INTEGER-EXACT: the device core mask must match sklearn's core set element by
    // element (core = self-inclusive eps-neighbor count >= min_samples).
    for i in 0..DB_N_SAMPLES {
        assert_eq!(
            mask.is_core[i], expected[i],
            "is_core[{i}] mismatch vs sklearn core_sample_indices_: \
             got={} expected={} (count={}, min_samples={min_samples})",
            mask.is_core[i], expected[i], mask.counts[i]
        );
    }
}

/// LOAD-NOT-JUST-PRESENT: the `dbscan` fixture loads with well-formed
/// X/eps/min_samples/labels/core_sample_indices arrays.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    assert_len(&case, "X", DB_N_SAMPLES * DB_N_FEATURES);
    assert_len(&case, "eps", 1);
    assert_len(&case, "min_samples", 1);
    assert_len(&case, "labels", DB_N_SAMPLES);
    let core = case.expect_f64("core_sample_indices");
    assert!(
        core.len() <= DB_N_SAMPLES,
        "core_sample_indices length {} must be <= n_samples {DB_N_SAMPLES}",
        core.len()
    );
}

/// Core-mask reproduces sklearn `core_sample_indices_`, f32 (runs on cpu AND
/// rocm). Integer-exact — it is a count threshold.
#[test]
fn dbscan_core_mask_matches_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_dbscan_mask::<f32>("dbscan_f32_seed42.npz");
}

/// Core-mask reproduces sklearn `core_sample_indices_`, f64 (cpu runs;
/// rocm skips-with-log). Integer-exact.
#[test]
fn dbscan_core_mask_matches_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("dbscan f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_dbscan_mask::<f64>("dbscan_f64_seed42.npz");
}

/// eps-neighborhood includes self (Pitfall 7): every point is its OWN eps-neighbor
/// (`D[i,i] = 0 <= eps²`), so `count[i] >= 1` for all `i`, and the self-bit on the
/// diagonal of the adjacency is set. f64 (cpu runs; rocm skips).
#[test]
fn dbscan_eps_neighborhood_includes_self_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    if capability::skip_f64_with_log() {
        println!("dbscan self-inclusivity f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let eps = case.expect_f64("eps")[0];
    let min_samples = case.expect_f64("min_samples")[0].round() as u32;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let mask = eps_core_mask::<f64>(
        &mut pool,
        &x_dev,
        DB_N_SAMPLES,
        DB_N_FEATURES,
        eps,
        min_samples,
    )
    .expect("eps_core_mask on valid geometry");
    x_dev.release_into(&mut pool);

    for i in 0..DB_N_SAMPLES {
        assert!(
            mask.counts[i] >= 1,
            "count[{i}] must be >= 1 (self-inclusive eps-neighborhood, Pitfall 7), got {}",
            mask.counts[i]
        );
        assert!(
            mask.adjacency[i * DB_N_SAMPLES + i],
            "adjacency[{i}][{i}] (self) must be set — D[i,i]=0 <= eps²"
        );
    }
}

/// Geometry / hyperparameter guard: a bad `x.len()`, a negative `eps`, and
/// `min_samples == 0` must each be rejected with `PrimError::ShapeMismatch` BEFORE
/// any launch (T-05-04-01 / ASVS V5), never an out-of-bounds device read. f32,
/// runs on cpu AND rocm.
#[test]
fn dbscan_rejects_bad_inputs() {
    use mlrs_core::PrimError;
    let _ = env_logger::builder().is_test(true).try_init();
    let n = 4usize;
    let d = 2usize;
    let x = vec![0.0f32; n * d];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    // n*d != x.len() → ShapeMismatch on "x".
    match eps_core_mask::<f32>(&mut pool, &x_dev, n + 1, d, 0.5, 2) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "x"),
        other => panic!("bad geometry must be ShapeMismatch on 'x', got {other:?}"),
    }
    // negative eps → ShapeMismatch on "eps".
    match eps_core_mask::<f32>(&mut pool, &x_dev, n, d, -1.0, 2) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "eps"),
        other => panic!("negative eps must be ShapeMismatch on 'eps', got {other:?}"),
    }
    // min_samples = 0 → ShapeMismatch on "min_samples".
    match eps_core_mask::<f32>(&mut pool, &x_dev, n, d, 0.5, 0) {
        Err(PrimError::ShapeMismatch { operand, .. }) => assert_eq!(operand, "min_samples"),
        other => panic!("min_samples=0 must be ShapeMismatch on 'min_samples', got {other:?}"),
    }

    x_dev.release_into(&mut pool);
}
