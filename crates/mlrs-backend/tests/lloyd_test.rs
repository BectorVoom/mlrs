//! Plan 05-03 — Lloyd centroid-update + inertia primitive standalone oracle.
//!
//! Exercises the NEW `mlrs_backend::prims::kmeans::{lloyd_update, inertia}` over
//! the committed `kmeans_{f32,f64}_seed42.npz` sklearn fixture: taking `X` + the
//! converged `labels`, `lloyd_update` must reproduce sklearn's `cluster_centers_`
//! (the per-label means) within 1e-5, and `inertia` over those centers+labels
//! must reproduce sklearn's `inertia_` (Σ squared distance to the assigned
//! center) within 1e-5. Runs STANDALONE before the KMeans estimator (07) consumes
//! it — the D-01 primitive-first discipline.
//!
//! A `lloyd_relocates_empty_cluster` case feeds a CONSTRUCTED assignment that
//! leaves one cluster id unused and asserts the empty cluster is RELOCATED to a
//! real data point (never a divide-by-zero NaN — T-05-03-02).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips, D-07). Per AGENTS.md §2 tests live here, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kmeans::{inertia, inertia_rows_host, lloyd_update};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// KMeans fixture geometry (gen_oracle.py KM_N_SAMPLES × KM_N_FEATURES, K=KM_K).
const KM_N_SAMPLES: usize = 30;
const KM_N_FEATURES: usize = 4;
const KM_K: usize = 3;

/// The project 1e-5 contract (centroids + inertia vs sklearn).
const TOL: f64 = 1e-5;

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("lloyd tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("lloyd tests are f32/f64 only"),
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

/// Shared oracle body: run `lloyd_update` from the fixture labels and assert the
/// centers match sklearn `cluster_centers_` within 1e-5; run `inertia` and assert
/// it matches sklearn `inertia_` within 1e-5. The fixture labels ARE sklearn's
/// converged assignment, so the per-label means equal `cluster_centers_` (sklearn
/// has run Lloyd to convergence) — no label-permutation ambiguity.
fn check_lloyd<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load kmeans fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let ref_centers: Vec<f64> = case.expect_f64("centers").to_vec(); // KM_K × KM_N_FEATURES
    let ref_inertia: f64 = case.expect_f64("inertia")[0];
    let labels: Vec<u32> = case
        .expect_f64("labels")
        .iter()
        .map(|&l| l.round() as u32)
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);

    // Per-sample distance to the assigned center (CR-01 relocation input). With
    // this balanced fixture no cluster empties, so it is never consulted, but pass
    // a correct slice (distances to the sklearn reference centers under labels).
    let ref_centers_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(&mut pool, &fixture_vec::<F>(&case, "centers"));
    let dist_to_assigned =
        inertia_rows_host::<F>(&mut pool, &x_dev, &ref_centers_dev, &labels, KM_N_SAMPLES, KM_N_FEATURES)
            .expect("inertia_rows_host on valid geometry");
    ref_centers_dev.release_into(&mut pool);

    // --- lloyd_update: per-label means must equal sklearn cluster_centers_. ---
    let centers_dev = lloyd_update::<F>(
        &mut pool,
        &x_dev,
        &labels,
        &dist_to_assigned,
        KM_N_SAMPLES,
        KM_N_FEATURES,
        KM_K,
    )
    .expect("lloyd_update on valid geometry");
    let got_centers: Vec<f64> = centers_dev
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();

    for c in 0..KM_K {
        for j in 0..KM_N_FEATURES {
            let slot = c * KM_N_FEATURES + j;
            let abs_err = (got_centers[slot] - ref_centers[slot]).abs();
            assert!(
                abs_err <= TOL + TOL * ref_centers[slot].abs(),
                "centroid[{c}][{j}] mismatch vs sklearn: got={:e} expected={:e} abs_err={abs_err:e}",
                got_centers[slot],
                ref_centers[slot]
            );
        }
    }

    // --- inertia: Σ squared distance to the assigned center vs sklearn inertia_. ---
    let got_inertia = host_to_f64(
        inertia::<F>(
            &mut pool,
            &x_dev,
            &centers_dev,
            &labels,
            KM_N_SAMPLES,
            KM_N_FEATURES,
        )
        .expect("inertia on valid geometry"),
    );
    let abs_err = (got_inertia - ref_inertia).abs();
    assert!(
        abs_err <= TOL + TOL * ref_inertia.abs(),
        "inertia mismatch vs sklearn: got={got_inertia:e} expected={ref_inertia:e} abs_err={abs_err:e}"
    );

    centers_dev.release_into(&mut pool);
    x_dev.release_into(&mut pool);
}

/// LOAD-NOT-JUST-PRESENT: the `kmeans` fixture loads with the injected `init`
/// (D-09) and well-formed centers/labels/inertia shapes.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "init", KM_K * KM_N_FEATURES);
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
    assert_len(&case, "labels", KM_N_SAMPLES);
    assert_len(&case, "inertia", 1);
}

/// Centroid sum-by-label reproduces sklearn `cluster_centers_` + inertia matches
/// `inertia_`, f32 (runs on cpu AND rocm).
#[test]
fn lloyd_centers_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_lloyd::<f32>("kmeans_f32_seed42.npz");
}

/// Centroids + inertia (Σ d²) reproduce sklearn, f64 (cpu runs; rocm skips).
#[test]
fn lloyd_inertia_matches_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("lloyd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_lloyd::<f64>("kmeans_f64_seed42.npz");
}

/// Empty-cluster relocation (T-05-03-02): a CONSTRUCTED assignment that uses only
/// clusters 0 and 1 (cluster 2 EMPTY) must NOT produce a NaN centroid for cluster
/// 2 — it is relocated to a real data point (a finite row of X). f32, runs on cpu
/// AND rocm.
#[test]
fn lloyd_relocates_empty_cluster() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    println!("lloyd empty-cluster relocation backend={backend} (T-05-03-02)");

    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");

    // Assign every sample to cluster 0 or 1 (alternating); cluster 2 is EMPTY.
    let labels: Vec<u32> = (0..KM_N_SAMPLES).map(|i| (i % 2) as u32).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    // Cluster 0/1 means (the "current" centers under `labels`), then each sample's
    // squared distance to its assigned center — the sklearn relocation input.
    let mut means = vec![0.0f64; KM_K * KM_N_FEATURES];
    let mut counts = [0u32; KM_K];
    for i in 0..KM_N_SAMPLES {
        let c = labels[i] as usize;
        counts[c] += 1;
        for j in 0..KM_N_FEATURES {
            means[c * KM_N_FEATURES + j] += x[i * KM_N_FEATURES + j] as f64;
        }
    }
    for c in 0..KM_K {
        if counts[c] > 0 {
            for j in 0..KM_N_FEATURES {
                means[c * KM_N_FEATURES + j] /= counts[c] as f64;
            }
        }
    }
    let means_f32: Vec<f32> = means.iter().map(|&v| v as f32).collect();
    let means_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &means_f32);
    let dist_to_assigned = inertia_rows_host::<f32>(
        &mut pool,
        &x_dev,
        &means_dev,
        &labels,
        KM_N_SAMPLES,
        KM_N_FEATURES,
    )
    .expect("inertia_rows_host on the cluster 0/1 means");
    means_dev.release_into(&mut pool);

    // The sklearn relocation target for the single empty cluster is the GLOBALLY
    // farthest sample (largest dist_to_assigned), lowest-index tie-break.
    let mut far_i = 0usize;
    let mut far_d = f64::NEG_INFINITY;
    for i in 0..KM_N_SAMPLES {
        if dist_to_assigned[i] > far_d {
            far_d = dist_to_assigned[i];
            far_i = i;
        }
    }

    let centers_dev = lloyd_update::<f32>(
        &mut pool,
        &x_dev,
        &labels,
        &dist_to_assigned,
        KM_N_SAMPLES,
        KM_N_FEATURES,
        KM_K,
    )
    .expect("lloyd_update with an empty cluster");
    let centers: Vec<f32> = centers_dev.to_host(&pool);
    centers_dev.release_into(&mut pool);
    x_dev.release_into(&mut pool);

    // Cluster 2's centroid (the relocated empty cluster) must be FINITE (no NaN /
    // no divide-by-zero).
    for j in 0..KM_N_FEATURES {
        let v = centers[2 * KM_N_FEATURES + j];
        assert!(
            v.is_finite(),
            "empty cluster 2 feature {j} must be finite after relocation, got {v}"
        );
    }
    // sklearn `_relocate_empty_clusters_dense`: the relocated centroid is EXACTLY
    // the globally-farthest-from-assigned-center sample row (CR-01).
    let relocated = &centers[2 * KM_N_FEATURES..3 * KM_N_FEATURES];
    for j in 0..KM_N_FEATURES {
        let expected = x[far_i * KM_N_FEATURES + j];
        assert!(
            (relocated[j] - expected).abs() <= 1e-6,
            "relocated empty cluster 2 feature {j} must be the farthest sample row {far_i}: got {}, expected {}",
            relocated[j],
            expected
        );
    }
}
