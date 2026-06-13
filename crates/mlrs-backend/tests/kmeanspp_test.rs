//! Plan 05-03 — k-means++ D²-weighted sampling primitive standalone INVARIANT
//! test.
//!
//! Exercises `mlrs_backend::prims::kmeans::kmeanspp_sample` (D-09a/c) — the
//! host-seeded D²-weighted default init. Because k-means++ uses RNG (D-09), there
//! is NO committed sklearn bit-for-bit reference; this is an INVARIANT test:
//!
//!   (a) the `k` sampled indices are DISTINCT and all in `0..n`;
//!   (b) running `kmeanspp_sample` TWICE with the SAME seed yields identical
//!       indices (seed-reproducibility, D-09c / T-05-03-03);
//!   (c) different seeds CAN differ (the draw actually consumes the RNG).
//!
//! Runs STANDALONE before the KMeans estimator (07) consumes it — the D-01
//! primitive-first discipline. The fixture supplies a real `X`; the D² weights
//! are device-computed (distance prim, sqrt=false) and the draw is a HOST PRNG
//! (SplitMix64 — never OsRng, ASVS V6).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips, D-07). Per AGENTS.md §2 tests live here, never an in-source
//! `#[cfg(test)] mod tests`.

use std::collections::HashSet;
use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kmeans::kmeanspp_sample;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// KMeans fixture geometry (gen_oracle.py KM_N_SAMPLES × KM_N_FEATURES, K=KM_K).
const KM_N_SAMPLES: usize = 30;
const KM_N_FEATURES: usize = 4;
const KM_K: usize = 3;

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("kmeanspp tests are f32/f64 only"),
    }
}

fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

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

/// Draw `k` k-means++ indices over the fixture `X` with the given seed.
fn sample<F>(fixture_name: &str, seed: u64) -> Vec<usize>
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load kmeans fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let chosen = kmeanspp_sample::<F>(
        &mut pool,
        &x_dev,
        KM_N_SAMPLES,
        KM_N_FEATURES,
        KM_K,
        seed,
    )
    .expect("kmeanspp_sample on valid geometry");
    x_dev.release_into(&mut pool);
    chosen
}

/// Assert the k sampled indices are DISTINCT and all in `0..n`.
fn assert_valid(chosen: &[usize]) {
    assert_eq!(chosen.len(), KM_K, "must sample exactly k={KM_K} centers");
    for &c in chosen {
        assert!(c < KM_N_SAMPLES, "sampled index {c} out of range 0..{KM_N_SAMPLES}");
    }
    let distinct: HashSet<usize> = chosen.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        KM_K,
        "k-means++ must sample DISTINCT centers, got {chosen:?}"
    );
}

/// LOAD-NOT-JUST-PRESENT: the `kmeans` fixture loads with its INJECTED `init` and
/// well-formed X/centers/labels/inertia shapes.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "init", KM_K * KM_N_FEATURES);
    assert_len(&case, "X", KM_N_SAMPLES * KM_N_FEATURES);
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
    assert_len(&case, "labels", KM_N_SAMPLES);
    assert_len(&case, "inertia", 1);
}

/// k-means++ draws VALID (distinct, in-range) centers and is SEED-REPRODUCIBLE,
/// f32 (runs on cpu AND rocm). Same seed → identical indices; a different seed is
/// allowed to differ (asserts the RNG is actually consumed).
#[test]
fn kmeanspp_valid_and_reproducible_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let a = sample::<f32>("kmeans_f32_seed42.npz", 42);
    assert_valid(&a);

    // (b) same seed → identical indices (seed-reproducibility, D-09c).
    let a2 = sample::<f32>("kmeans_f32_seed42.npz", 42);
    assert_eq!(
        a, a2,
        "same seed must yield identical k-means++ indices (seed-reproducibility)"
    );

    // (c) a different seed CAN differ — over several seeds at least one must
    // produce a different draw (proves the host RNG actually drives the sample;
    // a constant/ignored seed would make every draw identical).
    let differs = (0u64..8)
        .map(|s| sample::<f32>("kmeans_f32_seed42.npz", 100 + s))
        .any(|other| other != a);
    assert!(
        differs,
        "k-means++ must depend on the seed: no alternate seed changed the draw"
    );
}

/// k-means++ valid + seed-reproducible, f64 (cpu runs; rocm skips-with-log).
#[test]
fn kmeanspp_seed_reproducible_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kmeanspp f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let a = sample::<f64>("kmeans_f64_seed42.npz", 7);
    assert_valid(&a);
    let a2 = sample::<f64>("kmeans_f64_seed42.npz", 7);
    assert_eq!(
        a, a2,
        "same seed must yield identical k-means++ indices on f64 (seed-reproducibility)"
    );
}
