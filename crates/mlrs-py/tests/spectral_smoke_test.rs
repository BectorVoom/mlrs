//! Plan 09-04 — SpectralEmbedding / SpectralClustering construction + device
//! fit/accessor smoke (PY-06 incremental share).
//!
//! Rust **integration test** (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). Two parts:
//!
//!   - `spectral_estimators_construct_unfit` (runs today) — builds both wrappers
//!     via the Rust-callable `unfit_default()` and asserts they land in the
//!     `Unfit` arm, proving the two `any_estimator!`-generated enums + the two
//!     `#[pyclass]` definitions COMPILE and INSTANTIATE without a Python
//!     interpreter or a live device.
//!   - `spectral_fit_accessors` (un-ignored by 09-04) — the f32 + f64 device
//!     `fit` → `embedding_` / `labels_` accessor smoke. The PyO3 `fit` method is
//!     a thin `py.detach` shell over the same `mlrs_algos` estimators the
//!     wrappers delegate to (`SpectralEmbedding<F>` / `SpectralClustering<F>`),
//!     so this drives those algos `fit` bodies directly on a live device (no
//!     Python interpreter needed at the Rust test level — the full
//!     interpreter+capsule path runs in the pytest harness, the 08-05 kernel
//!     precedent). f64 is gated by `capability::skip_f64_with_log()` (skip on a
//!     backend without f64, e.g. rocm; run on cpu).

use mlrs_algos::cluster::{SpectralClustering, SpectralEmbedding};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

use mlrs_py::estimators::spectral::{PySpectralClustering, PySpectralEmbedding};

/// A tiny well-separated `n × d` design so the embedding + clustering fit lands a
/// stable partition (geometry only — the value gates live in the algos oracle
/// tests; this smoke just proves the fit + accessor surface runs end to end).
const N: usize = 6;
const D: usize = 2;

fn x_host<F>() -> Vec<F>
where
    F: bytemuck::Pod + cubecl::prelude::Float,
{
    // Two well-separated triples (centers far apart) → a clean 2-cluster split.
    let rows: [[f64; D]; N] = [
        [0.0, 0.0],
        [0.1, -0.1],
        [-0.1, 0.1],
        [10.0, 10.0],
        [10.1, 9.9],
        [9.9, 10.1],
    ];
    let mut v = Vec::with_capacity(N * D);
    for r in &rows {
        for &c in r {
            v.push(match std::mem::size_of::<F>() {
                4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(c as f32))),
                8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&c)),
                _ => unreachable!("smoke design is f32/f64 only"),
            });
        }
    }
    v
}

/// Both spectral wrappers construct with default hyperparameters and start
/// `Unfit` (no Python interpreter / live device needed).
#[test]
fn spectral_estimators_construct_unfit() {
    assert!(
        PySpectralEmbedding::unfit_default().is_unfit(),
        "SpectralEmbedding"
    );
    assert!(
        PySpectralClustering::unfit_default().is_unfit(),
        "SpectralClustering"
    );
}

/// Device fit/accessor smoke for both spectral estimators (the algos bodies the
/// PyO3 `fit` shells delegate to), f32 always + f64 when `skip_f64_with_log()` is
/// false. SpectralEmbedding → `embedding_` (n × n_components); SpectralClustering
/// → `labels_` (length n, i32).
#[test]
fn spectral_fit_accessors() {
    let _ = env_logger::builder().is_test(true).try_init();

    // The wrapper construction surface is proven by the test above.
    assert!(PySpectralEmbedding::unfit_default().is_unfit());
    assert!(PySpectralClustering::unfit_default().is_unfit());

    // f32 always.
    fit_embedding_smoke::<f32>();
    fit_clustering_smoke::<f32>();

    // f64 gated by backend capability (skip on rocm, run on cpu).
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "smoke");
    if capability::skip_f64_with_log() {
        println!("spectral_fit_accessors f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    fit_embedding_smoke::<f64>();
    fit_clustering_smoke::<f64>();
}

/// SpectralEmbedding `fit` → `embedding_` accessor smoke: rbf affinity, 2 dims.
fn fit_embedding_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xh = x_host::<F>();
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xh);

    let n_components = 2usize;
    let mut se = SpectralEmbedding::<F>::new(n_components, "rbf".to_string(), None, 3);
    se.fit(&mut pool, &xd, (N, D)).expect("SpectralEmbedding::fit smoke");
    let emb = se.embedding(&pool).expect("embedding_ after fit");
    assert_eq!(emb.len(), N * n_components, "embedding_ is n × n_components");
    assert!(
        emb.iter().all(|v| {
            let f = match std::mem::size_of::<F>() {
                4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
                8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
                _ => unreachable!(),
            };
            f.is_finite()
        }),
        "embedding_ is all-finite"
    );
}

/// SpectralClustering `fit` → `labels_` accessor smoke: rbf affinity, k=2 → a
/// clean 2-way split of the well-separated design.
fn fit_clustering_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xh = x_host::<F>();
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xh);

    let gamma: F = match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&1.0f32)),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&1.0f64)),
        _ => unreachable!(),
    };
    let mut sc = SpectralClustering::<F>::new(2, None, "rbf".to_string(), gamma, 3, 7);
    sc.fit(&mut pool, &xd, (N, D)).expect("SpectralClustering::fit smoke");
    let labels = sc.labels(&pool).expect("labels_ after fit");
    assert_eq!(labels.len(), N, "labels_ is length n");
    assert!(
        labels.iter().all(|&l| l >= 0 && (l as usize) < 2),
        "labels_ are in 0..k"
    );
    // The two well-separated triples must NOT share a cluster (a clean split).
    assert_eq!(labels[0], labels[1], "first triple is one cluster");
    assert_eq!(labels[1], labels[2], "first triple is one cluster");
    assert_eq!(labels[3], labels[4], "second triple is one cluster");
    assert_ne!(labels[0], labels[3], "the two triples split into 2 clusters");
}
