//! Plan 12-02 Wave-2 — UMAP shell (UMAP-01) convention tests.
//!
//! These exercise the Phase-12 builder + typestate CONVENTION end-to-end on the
//! `Umap<F, S = Unfit>` shell (no algorithm — the fit emits a zeros embedding):
//!
//!   - `defaults_equal` — `Umap::new()` and `Umap::builder().build::<F>()?`
//!     agree on the hyperparameter subset (single-source defaults, BLDR-01).
//!   - `build_rejects_bad_min_dist` — `min_dist > spread` is rejected at
//!     `build()` with `BuildError::InvalidMinDist` (the D-08 data-independent
//!     split — never at fit).
//!   - `fit_roundtrip` — the trivial fit produces an all-zeros embedding of
//!     length `n * n_components`, and `n_features_in()` reports `p` (D-10).
//!   - `fit_no_leak` — re-CONSTRUCT + re-fit at the same shape does not grow
//!     `live_bytes` (the consuming fit forces reconstruction each iteration).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate (cpu runs f64; rocm
//! skips). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use mlrs_algos::error::BuildError;
use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// BLDR-01: `Umap::new()` equals `Umap::builder().build()?` on the
/// hyperparameter subset (single-source defaults). Pure host comparison — no
/// device, so no f64 gate needed.
#[test]
fn defaults_equal() {
    let from_new = Umap::<f64>::new();
    let from_builder = Umap::<f64>::builder()
        .build::<f64>()
        .expect("default UmapBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Umap::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// D-08 / T-12-02: a `min_dist > spread` is rejected at `build()` with the typed
/// `BuildError::InvalidMinDist`, BEFORE any data is seen.
#[test]
fn build_rejects_bad_min_dist() {
    let bad = Umap::<f64>::builder()
        .min_dist(2.0)
        .spread(1.0)
        .build::<f64>()
        .err();
    assert!(
        matches!(
            bad,
            Some(BuildError::InvalidMinDist { min_dist, .. }) if min_dist == 2.0
        ),
        "min_dist > spread must be BuildError::InvalidMinDist, got {bad:?}"
    );
}

/// D-10 runtime proof: the trivial fit round-trips — `embedding()` returns
/// `n * n_components` zeros and `n_features_in()` reports `p`.
#[test]
fn fit_roundtrip() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("umap fit_roundtrip f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 6usize;
    let p = 3usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Umap::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("trivial fit succeeds");

    let n_components = 2usize; // default
    let embedding = fitted.embedding(&pool);
    assert_eq!(
        embedding.len(),
        n * n_components,
        "embedding length must be n * n_components"
    );
    assert!(
        embedding.iter().all(|&v| v == 0.0),
        "trivial fit embedding must be all zeros"
    );
    assert_eq!(
        fitted.n_features_in(),
        p,
        "n_features_in() must report the fit-time feature count"
    );
}

/// Memory gate: re-CONSTRUCT + re-fit at the same shape does not grow
/// `live_bytes`. The consuming `fit(self)` means each iteration rebuilds the
/// estimator rather than re-fitting one instance.
#[test]
fn fit_no_leak() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("umap fit_no_leak f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 8usize;
    let p = 4usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    // Warm up: first construct + fit; record steady live_bytes.
    let fitted = Umap::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    const REFITS: usize = 4;
    for k in 0..REFITS {
        let fitted = Umap::<f64>::new()
            .fit(&mut pool, &x_dev, None, (n, p))
            .expect("re-fit");
        drop(fitted);
        let live = pool.stats().live_bytes;
        assert!(
            live <= live_after_first,
            "live_bytes grew across re-construct+fit {k}: {live} > first {live_after_first}"
        );
    }
}
