//! Plan 12-02 Wave-2 — HDBSCAN shell (HDBS-01) convention tests.
//!
//! These exercise the Phase-12 builder + typestate CONVENTION end-to-end on the
//! `Hdbscan<F, S = Unfit>` shell (no algorithm — the fit emits all-`-1` labels):
//!
//!   - `defaults_equal` — `Hdbscan::new()` and `Hdbscan::builder().build()?`
//!     agree on the hyperparameter subset (single-source defaults, BLDR-01).
//!   - `build_rejects_bad_min_cluster_size` — `min_cluster_size < 2` is rejected
//!     at `build()` with `BuildError::InvalidMinClusterSize` (the D-08
//!     data-independent split — never at fit).
//!   - `fit_roundtrip` — the trivial fit produces an all-`-1` labels vector of
//!     length `n`, and `n_features_in()` reports `p` (D-10).
//!   - `fit_no_leak` — re-CONSTRUCT + re-fit at the same shape does not grow
//!     `live_bytes` (the consuming fit forces reconstruction each iteration).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate. Per AGENTS.md §2
//! tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_algos::cluster::hdbscan::Hdbscan;
use mlrs_algos::error::BuildError;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// BLDR-01: `Hdbscan::new()` equals `Hdbscan::builder().build()?` on the
/// hyperparameter subset (single-source defaults). Pure host comparison.
#[test]
fn defaults_equal() {
    let from_new = Hdbscan::<f64>::new();
    let from_builder = Hdbscan::<f64>::builder()
        .build::<f64>()
        .expect("default HdbscanBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Hdbscan::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// D-08 / T-12-02: a `min_cluster_size < 2` is rejected at `build()` with the
/// typed `BuildError::InvalidMinClusterSize`, BEFORE any data is seen.
#[test]
fn build_rejects_bad_min_cluster_size() {
    let bad = Hdbscan::<f64>::builder()
        .min_cluster_size(1)
        .build::<f64>()
        .err();
    assert!(
        matches!(
            bad,
            Some(BuildError::InvalidMinClusterSize { min_cluster_size, .. })
                if min_cluster_size == 1
        ),
        "min_cluster_size < 2 must be BuildError::InvalidMinClusterSize, got {bad:?}"
    );
}

/// D-10 runtime proof: the trivial fit round-trips — `labels()` returns `n`
/// all-`-1` entries and `n_features_in()` reports `p`.
#[test]
fn fit_roundtrip() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("hdbscan fit_roundtrip f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 7usize;
    let p = 3usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Hdbscan::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("trivial fit succeeds");

    let labels = fitted.labels(&pool);
    assert_eq!(labels.len(), n, "labels length must be n");
    assert!(
        labels.iter().all(|&v| v == -1),
        "trivial fit labels must be all -1 (noise sentinel)"
    );
    assert_eq!(
        fitted.n_features_in(),
        p,
        "n_features_in() must report the fit-time feature count"
    );
}

/// Memory gate: re-CONSTRUCT + re-fit at the same shape does not grow
/// `live_bytes`. The consuming `fit(self)` forces reconstruction each iteration.
#[test]
fn fit_no_leak() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("hdbscan fit_no_leak f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 9usize;
    let p = 4usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Hdbscan::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    const REFITS: usize = 4;
    for k in 0..REFITS {
        let fitted = Hdbscan::<f64>::new()
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
