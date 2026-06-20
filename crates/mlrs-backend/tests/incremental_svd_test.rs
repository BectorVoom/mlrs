//! Plan 07-03 — incremental (batched) SVD merge primitive (PRIM-07) tests.
//!
//! PRIM-07 gate: a 2+-batch incremental merge whose running `(components_,
//! singular_values_, explained_variance_, mean_, var_, n_samples_seen_)` is
//! compared against sklearn's own `IncrementalPCA` (the committed
//! `incremental_pca_*` oracle blob from plan 07-01). The merge replicates
//! sklearn's exact `partial_fit` stream (the fixture's `batch_size`, ≥3 batches
//! over 30 rows) and matches sklearn's `components_` (post `align_rows`),
//! `singular_values_`, `explained_variance_`, `mean_`, `var_`,
//! `n_samples_seen_`. f64 holds strict `F64_TOL` (1e-5); f32 uses a documented
//! per-family band measured on THIS standalone test (RESEARCH A3/A4). Plus a
//! PoolStats memory gate over a multi-batch stream.
//!
//! NOTE: IncrementalPCA is sklearn's own approximation — a multi-batch merge does
//! NOT equal a single full-matrix SVD of the whole design (the mean-correction
//! row makes them differ at >1e-5). The oracle is therefore sklearn's
//! IncrementalPCA attributes, NOT a single-pass SVD reference.
//!
//! The SVD merge is sized so the stacked matrix clears the Phase-3 caps
//! (`n_components + batch_size + 1 ≤ MAX_ROWS`, `n_features ≤ MAX_COLS`).
//! Fixtures are kept TINY (the SVD path is the slow one — cpu suite ~6 min).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim from
//! `gemm_test.rs` (cpu runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2
//! tests live in `crates/mlrs-backend/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::incremental_svd::{merge, IncrementalSvdState};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{assert_slice_close, load_npz, OracleCase, Tolerance, F64_TOL};

/// Documented f32 band for the standalone PRIM-07 merge (RESEARCH A3/A4 — set
/// FROM the measurement printed by `incremental_svd_two_batch_merge`; see the
/// SUMMARY for the observed max abs/rel error). The re-expanded `Σ·Vᵀ` preserves
/// the rank-k energy exactly so the per-batch error does not compound; the band
/// matches the v1 PCA f32 family band (1e-4) with margin.
const F32_MERGE_TOL: Tolerance = Tolerance::new(1e-4, 1e-4);

/// Resolve a workspace-root-relative fixture path (matches `svd_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Build an `F` (f32/f64) from an `f64`.
fn from_f64<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("incremental_svd is f32/f64 only"),
    }
}

/// Drive the multi-batch merge over the fixture's exact `batch_size` stream
/// (sklearn `fit()` loops `partial_fit` over `gen_batches(n, batch_size)`), then
/// assert the running state matches sklearn's committed IncrementalPCA attributes
/// within `tol` (post `align_rows` on both component sets).
fn run_oracle_case<F>(case: &OracleCase, n: usize, p: usize, nc: usize, bs: usize, tol: &Tolerance)
where
    F: Float + CubeElement + Pod,
{
    let x64: Vec<f64> = case.expect_f64("X").to_vec();
    let ref_sv: Vec<f64> = case.expect_f64("singular_values_").to_vec();
    let ref_ev: Vec<f64> = case.expect_f64("explained_variance_").to_vec();
    let ref_mean: Vec<f64> = case.expect_f64("mean_").to_vec();
    let ref_var: Vec<f64> = case.expect_f64("var_").to_vec();
    let ref_comp_flat: Vec<f64> = case.expect_f64("components_").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // --- Incremental: stream `gen_batches(n, bs)` exactly as sklearn fit(). ---
    let mut state: Option<IncrementalSvdState> = None;
    let mut start = 0usize;
    while start < n {
        let b = bs.min(n - start);
        let batch_f: Vec<F> = x64[start * p..(start + b) * p]
            .iter()
            .map(|&v| from_f64::<F>(v))
            .collect();
        let batch_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &batch_f);
        state = Some(
            merge::<F>(&mut pool, state.take(), &batch_dev, (b, p), nc).expect("merge batch"),
        );
        batch_dev.release_into(&mut pool);
        start += b;
    }
    let state = state.expect("at least one batch");

    // --- n_samples_seen_ exact; mean_/var_ match sklearn. ---
    assert_eq!(state.n_samples_seen_, n, "n_samples_seen_ == total");
    assert_slice_close(&state.mean_, &ref_mean, tol);
    assert_slice_close(&state.var_, &ref_var, tol);

    // --- singular_values_ + explained_variance_ match (post-truncation). ---
    assert_slice_close(&state.singular_values_, &ref_sv, tol);
    assert_slice_close(&state.explained_variance_, &ref_ev, tol);

    // --- components_ match per-row after align_rows on BOTH (sklearn's
    //     components_ are already svd_flip'd; re-align both for a sign-stable
    //     compare — Pitfall 5). ---
    let comp_host = state.components_.to_host(&mut pool);
    let merged_rows: Vec<Vec<f64>> = (0..nc)
        .map(|j| (0..p).map(|c| comp_host[j * p + c]).collect())
        .collect();
    let merged_aligned = align_rows(&merged_rows);
    let ref_rows: Vec<Vec<f64>> = (0..nc)
        .map(|j| (0..p).map(|c| ref_comp_flat[j * p + c]).collect())
        .collect();
    let ref_aligned = align_rows(&ref_rows);

    // Measure + report the observed max abs/rel error on this standalone test
    // (the band source for IncrementalPCA in Plan 05 — RESEARCH A3/A4).
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for j in 0..nc {
        for c in 0..p {
            let g = merged_aligned[j][c];
            let e = ref_aligned[j][c];
            let abs = (g - e).abs();
            max_abs = max_abs.max(abs);
            if e.abs() > 1e-8 {
                max_rel = max_rel.max(abs / e.abs());
            }
        }
    }
    let dtype = if std::mem::size_of::<F>() == 4 { "f32" } else { "f64" };
    println!(
        "incremental_svd_two_batch_merge[{dtype}] components_ max_abs={max_abs:e} max_rel={max_rel:e} \
         (tol.abs={:e} tol.rel={:e})",
        tol.abs, tol.rel
    );

    for j in 0..nc {
        assert_slice_close(&merged_aligned[j], &ref_aligned[j], tol);
    }
}

/// Load the committed sklearn IncrementalPCA oracle blob + its `(n, p, nc, bs)`.
fn load_oracle(name: &str) -> (OracleCase, usize, usize, usize, usize) {
    let case = load_npz(fixture(name)).unwrap_or_else(|_| panic!("load {name}"));
    let x_shape = case.shape("X").expect("X shape").to_vec();
    let n = x_shape[0] as usize;
    let p = x_shape[1] as usize;
    let nc = case.expect_f64("n_components")[0] as usize;
    let bs = case.expect_f64("batch_size")[0] as usize;
    (case, n, p, nc, bs)
}

/// 2+-batch incremental merge vs sklearn's committed IncrementalPCA attributes
/// (PRIM-07), f64 strict `F64_TOL`. Gated by `skip_f64_with_log` (cpu runs; rocm
/// skips). The fixture's `batch_size=10` over 30 rows is a 3-batch stream.
#[test]
fn incremental_svd_two_batch_merge() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_svd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let (case, n, p, nc, bs) = load_oracle("incremental_pca_nowhiten_f64_seed42.npz");
    run_oracle_case::<f64>(&case, n, p, nc, bs, &F64_TOL);
}

/// 2+-batch incremental merge vs sklearn's IncrementalPCA at the documented f32
/// band (`F32_MERGE_TOL`). Runs on every backend (the f32 gate is rocm; cpu also
/// exercises f32). The observed max abs/rel error is printed (band source).
#[test]
fn incremental_svd_two_batch_merge_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (case, n, p, nc, bs) = load_oracle("incremental_pca_nowhiten_f32_seed42.npz");
    run_oracle_case::<f32>(&case, n, p, nc, bs, &F32_MERGE_TOL);
}

/// PoolStats memory gate for `incremental_svd.rs` (PRIM-07): driving `merge` N
/// times at a fixed batch shape releases the per-batch SVD scratch + the uploaded
/// stack — `live_bytes` conserves after warmup and `peak_bytes` plateaus (the
/// D-10 one-gate-per-prim precedent). One svd + one stack upload per merge; no
/// leak across calls.
#[test]
fn incremental_svd_memory_gate() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    const N: usize = 5;
    let p = 6usize;
    let b = 6usize; // fixed batch shape; nc+b+1 = 3+6+1 = 10 ≤ MAX_ROWS.
    let nc = 3usize;

    // Deterministic batch data (the gate asserts on POOL COUNTERS, not values).
    let make_batch = |seed: usize| -> Vec<f32> {
        (0..b * p)
            .map(|i| (((i + seed) % 13) as f32) * 0.1 - 0.6)
            .collect()
    };

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let mut live_after: Vec<u64> = Vec::with_capacity(N);
    let mut peak_after: Vec<u64> = Vec::with_capacity(N);

    let mut state: Option<IncrementalSvdState> = None;
    for iter in 0..N {
        let batch = make_batch(iter);
        let batch_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &batch);
        state = Some(
            merge::<f32>(&mut pool, state.take(), &batch_dev, (b, p), nc)
                .expect("merge in memory gate"),
        );
        batch_dev.release_into(&mut pool);

        let stats = pool.stats();
        live_after.push(stats.live_bytes);
        peak_after.push(stats.peak_bytes);
    }

    // Drop the final running state's device component back to the pool.
    if let Some(s) = state {
        s.components_.release_into(&mut pool);
    }

    // After a warmup iteration the live footprint must CONSERVE (the per-batch
    // SVD scratch + uploaded stack are released; only the carried device
    // `components_` persists — a fixed, bounded footprint). A monotone climb is
    // the RED-if-removed signal that a release went missing.
    for w in 2..N {
        assert!(
            live_after[w] <= live_after[1],
            "live_bytes must not grow after warmup: iter {w} = {} > iter 1 = {}",
            live_after[w],
            live_after[1]
        );
    }
    // peak_bytes plateaus after warmup (released scratch reused in place).
    for w in 2..N {
        assert_eq!(
            peak_after[w], peak_after[N - 1],
            "peak_bytes must plateau after warmup (iter {w} vs final)"
        );
    }

    println!(
        "incremental_svd_memory_gate backend={backend}: live={:?} peak={:?} (N={N})",
        live_after, peak_after
    );
}
