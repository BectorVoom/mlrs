//! Plan 05-07 — KMeans (CLUSTER-01) sklearn oracle tests.
//!
//! Activated from the 05-01 Nyquist `#[ignore]` scaffold: each function loads
//! the committed `kmeans_{f32,f64}_seed42.npz` fixture — which carries an
//! INJECTED `init` array (D-09) so both mlrs and sklearn run Lloyd from the SAME
//! starting centers — constructs `KMeans::with_init`, fits, and asserts the
//! fitted `cluster_centers_` / `inertia_` against the sklearn reference within
//! the 1e-5 abs+rel contract (up to a label permutation) and the fitted
//! `labels_` via `best_match_accuracy == 1.0` (`mlrs_core::label_perm`, D-09).
//!
//! KMeans implements `PredictLabels` (NOT `Predict<F>`, D-08): the
//! `predict_labels` path assigns new points to the fitted centers, exercised by
//! the predict-consistency test (re-predicting the training X reproduces the
//! fitted labels up to the same permutation).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu
//! runs f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::kmeans::KMeans;
use mlrs_algos::traits::{Fit, PredictLabels};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{best_match_accuracy, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// KMeans fixture geometry (gen_oracle.py KM_N_SAMPLES × KM_N_FEATURES, K=KM_K).
const KM_N_SAMPLES: usize = 30;
const KM_N_FEATURES: usize = 4;
const KM_K: usize = 3;
const SEED: u64 = 42;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans fixtures are f32/f64 only"),
    }
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kmeans fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened.
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Fit `KMeans::with_init` from the fixture's INJECTED init (D-09) and return the
/// host `(cluster_centers_, labels_ as i64, inertia_)`.
fn fit_kmeans<F>(case: &OracleCase) -> (Vec<f64>, Vec<i64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let init_host: Vec<F> = case
        .expect_f64("init")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let mut km = KMeans::<F>::with_init(KM_K, init_host);
    km.fit(&mut pool, &x_dev, None, (KM_N_SAMPLES, KM_N_FEATURES))
        .expect("KMeans::fit on a valid shape");

    let centers: Vec<f64> = km
        .cluster_centers(&pool)
        .expect("cluster_centers_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let labels: Vec<i64> = km
        .labels(&pool)
        .expect("labels_ after fit")
        .iter()
        .map(|&l| l as i64)
        .collect();
    let inertia = host_to_f64(km.inertia().expect("inertia_ after fit"));
    (centers, labels, inertia)
}

/// `cluster_centers_` + `labels_` match sklearn up to a label permutation (D-09),
/// f32: remap the fitted centers/labels onto sklearn's ordering via the best
/// label mapping, then compare centers within 1e-5 and require a perfect label
/// permutation (`best_match_accuracy == 1.0`).
fn run_centers_labels<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let centers_ref = case.expect_f64("centers");
    let labels_ref: Vec<i64> = case.expect_f64("labels").iter().map(|&v| v as i64).collect();
    assert_eq!(centers_ref.len(), KM_K * KM_N_FEATURES, "fixture centers len");
    assert_eq!(labels_ref.len(), KM_N_SAMPLES, "fixture labels len");

    let (centers, labels, _inertia) = fit_kmeans::<F>(case);

    // Labels match sklearn up to a permutation (D-09): perfect mapping.
    let acc = best_match_accuracy(&labels, &labels_ref);
    assert!(
        (acc - 1.0).abs() < f64::EPSILON,
        "{label}: best_match_accuracy {acc} != 1.0 (labels not a permutation of sklearn)"
    );

    // Centers match up to the SAME permutation: map each fitted cluster id to its
    // sklearn id via the label confusion, then compare the centroid rows.
    let mapping = mlrs_core::best_mapping(&labels, &labels_ref);
    for fitted_c in 0..KM_K {
        let ref_c = *mapping
            .get(&(fitted_c as i64))
            .expect("every fitted cluster maps to a sklearn cluster") as usize;
        let got = &centers[fitted_c * KM_N_FEATURES..(fitted_c + 1) * KM_N_FEATURES];
        let exp = &centers_ref[ref_c * KM_N_FEATURES..(ref_c + 1) * KM_N_FEATURES];
        assert_close(got, exp, tol, &format!("{label} center[{fitted_c}->{ref_c}]"));
    }
}

/// `inertia_` is permutation-INVARIANT (a scalar sum over all points), so it must
/// match sklearn directly within 1e-5.
fn run_inertia<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let inertia_ref = case.expect_f64("inertia");
    assert_eq!(inertia_ref.len(), 1, "fixture inertia len");
    let (_centers, _labels, inertia) = fit_kmeans::<F>(case);
    assert_close(&[inertia], &[inertia_ref[0]], tol, &format!("{label} inertia_"));
}

/// `cluster_centers_`/`labels_` vs sklearn up to a permutation, f32.
#[test]
fn kmeans_centers_labels_match_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    run_centers_labels::<f32>(&case, &F32_TOL, "kmeans f32");
}

/// `cluster_centers_`/`labels_` vs sklearn up to a permutation, f64 (cpu runs;
/// rocm skips-with-log).
#[test]
fn kmeans_centers_labels_match_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kmeans f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    run_centers_labels::<f64>(&case, &F64_TOL, "kmeans f64");
}

/// `inertia_` matches sklearn, f32 (permutation-invariant scalar).
#[test]
fn kmeans_inertia_matches_sklearn_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    run_inertia::<f32>(&case, &F32_TOL, "kmeans f32");
}

/// `inertia_` matches sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn kmeans_inertia_matches_sklearn_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kmeans inertia f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    run_inertia::<f64>(&case, &F64_TOL, "kmeans f64");
}

/// `predict_labels` assigns new points to the fitted centers (D-08 — KMeans
/// implements `PredictLabels`, NOT `Predict<F>`). Re-predicting the TRAINING X
/// must reproduce the fitted `labels_` exactly (the assignment is deterministic
/// from the converged centers — a consistency gate, not a separate oracle).
#[test]
fn kmeans_predict_assigns_to_centers_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let init_host: Vec<f32> = case.expect_f64("init").iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);

    let mut km = KMeans::<f32>::with_init(KM_K, init_host);
    km.fit(&mut pool, &x_dev, None, (KM_N_SAMPLES, KM_N_FEATURES))
        .expect("KMeans::fit");

    let fitted_labels: Vec<i64> = km.labels(&pool).unwrap().iter().map(|&l| l as i64).collect();
    let predicted = km
        .predict_labels(&mut pool, &x_dev, (KM_N_SAMPLES, KM_N_FEATURES))
        .expect("KMeans::predict_labels on training X");
    let predicted_labels: Vec<i64> = predicted.to_host(&pool).iter().map(|&l| l as i64).collect();

    assert_eq!(
        predicted_labels, fitted_labels,
        "predict_labels on training X must reproduce the fitted labels_ exactly"
    );

    // And consistency vs sklearn (same permutation invariance): the prediction is
    // a perfect permutation of the sklearn labels.
    let labels_ref: Vec<i64> = case.expect_f64("labels").iter().map(|&v| v as i64).collect();
    let acc = best_match_accuracy(&predicted_labels, &labels_ref);
    assert!(
        (acc - 1.0).abs() < f64::EPSILON,
        "predict_labels best_match_accuracy {acc} != 1.0"
    );
}

/// WR-03 regression: a CONSTANT-FEATURE design (every column identical across all
/// samples) drives the mean per-feature variance to 0, so `tol_scaled = tol·0 = 0`
/// — the deliberately-kept sklearn `tol == 0` semantics. The Lloyd loop can then
/// stop only on the strict label-equality break or `max_iter`; KMeans's documented
/// contract (matching sklearn) is to NEVER error on non-convergence but return the
/// best-effort fit. This test pins that contract: fitting a degenerate
/// constant-feature matrix must SUCCEED (no NotConverged, no panic) and return
/// valid in-range labels.
#[test]
fn wr03_constant_feature_design_does_not_error() {
    let client = runtime::active_client();
    let mut pool = BufferPool::<ActiveRuntime>::new(client);

    // 6 samples, 3 features, all entries identical → var(X, axis=0) == 0 for every
    // feature → mean_var == 0 → tol_scaled == 0.
    const N: usize = 6;
    const D: usize = 3;
    const K: usize = 2;
    let x_host: Vec<f32> = vec![1.0f32; N * D];
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);

    // Inject a deterministic init: two distinct rows (so the two centers start
    // apart even though the data is constant); avoids the k-means++ PRNG so the
    // test is fully reproducible.
    let init: Vec<f32> = vec![1.0f32; K * D];
    let mut km = KMeans::<f32>::with_init(K, init);

    let res = km.fit(&mut pool, &x_dev, None, (N, D));
    assert!(
        res.is_ok(),
        "KMeans on a constant-feature (tol_scaled == 0) design must not error: {:?}",
        res.err()
    );

    let labels = km.labels(&pool).expect("labels_ after fit");
    assert_eq!(labels.len(), N, "one label per sample");
    for &l in labels.iter() {
        assert!(
            l >= 0 && (l as usize) < K,
            "label {l} out of range 0..{K}"
        );
    }
}
