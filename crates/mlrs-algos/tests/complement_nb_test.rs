//! Plan 11-03 Wave-1 — ComplementNB (NB-04) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The estimator fits
//! `feature_count_` via the validated `class_grouped_sum` GATHER, forms the
//! complement counts `comp[c,j] = feature_all_[j] + alpha − feature_count_[c,j]`,
//! the `logged[c,j] = log(comp[c,j] / Σ_j comp[c,j])`, and the sklearn weights
//! `feature_log_prob_ = −logged` (default) or `logged / Σ_j logged` (norm)
//! (Pitfall 6 — a DIFFERENT formula from MultinomialNB). The joint LL is
//! `X @ feature_log_prob_.T`; labels decode with `argmin` over `−jll` (D-08),
//! proba log-sum-exp-normalizes `jll`:
//!
//!   - `exact_labels` / `exact_labels_f32` — predict labels match sklearn EXACTLY
//!     (the argmin convention is correct, NOT sign-flipped).
//!   - `proba_band` — predict_proba within band + rows sum to 1.0 ± 1e-6.
//!   - `default_matches_sklearn` — bare builder reproduces sklearn (norm=false).
//!   - `norm_true` — the second L1 normalization yields weight rows summing to 1
//!     and valid (rows-sum-to-1) proba.
//!   - `build_rejects_bad_alpha` — `build()` rejects `alpha < 0`.
//!   - `refit_releases_buffers` — the PoolStats no-leak gate across a re-fit.
//!
//! f64 cases carry the `skip_f64_with_log` gate (D-07). Per AGENTS.md §2 tests
//! live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::naive_bayes::ComplementNB;
// Phase 16 (D-02): ComplementNB migrated to the typestate surface — consuming-self
// `Fit` + `Fitted`-gated accessors consumed via UFCS through these aliases.
use mlrs_algos::typestate::{
    Fit as TypestateFit, PredictLabels as TypestatePredictLabels,
    PredictProba as TypestatePredictProba,
};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

const N_SAMPLES: usize = 39;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 6;
const N_CLASSES: usize = 3;

const PROBA_BAND_F64: f64 = 1e-5;
const PROBA_BAND_F32: f64 = 1e-3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("complement_nb fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("complement_nb fixtures are f32/f64 only"),
    }
}

fn assert_band(got: &[f64], expected: &[f64], band: f64, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= band + band * e.abs(),
            "{what}: band failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e} (band={band:e})"
        );
    }
}

fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(case.expect_f64("X").len(), N_SAMPLES * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("predict").len(), N_QUERY);
    assert_eq!(case.expect_f64("predict_proba").len(), N_QUERY * N_CLASSES);
}

fn assert_rows_sum_to_one(proba: &[f64]) {
    for (r, row) in proba.chunks(N_CLASSES).enumerate() {
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() <= 1e-6,
            "predict_proba row {r} sums to {s} (expected 1.0 ± 1e-6)"
        );
    }
}

/// Build (sklearn defaults: norm=false) + fit a `ComplementNB` and return host
/// `(predict_labels(Xq), predict_proba(Xq))`.
fn fit_complement<F>(case: &OracleCase) -> (Vec<i32>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = case.expect_f64("y").iter().map(|&v| f64_to::<F>(v)).collect();
    let xq_host: Vec<F> = case.expect_f64("Xq").iter().map(|&v| f64_to::<F>(v)).collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq_host);

    let clf = ComplementNB::<F>::builder()
        .build::<F>()
        .expect("default ComplementNB builds");
    let clf = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("ComplementNB::fit on a valid shape");

    let labels =
        TypestatePredictLabels::predict_labels(&clf, &mut pool, &xq_dev, (N_QUERY, N_FEATURES))
            .expect("predict_labels after fit")
            .to_host(&pool);
    let proba: Vec<f64> =
        TypestatePredictProba::predict_proba(&clf, &mut pool, &xq_dev, (N_QUERY, N_FEATURES))
            .expect("predict_proba after fit")
            .to_host(&pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();

    (labels, proba)
}

/// HARD GATE: predict labels match sklearn EXACTLY (argmin convention), f32.
#[test]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("complement_nb_f32_seed42.npz")).expect("load complement_nb_f32");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case.expect_f64("predict").iter().map(|&v| v.round() as i32).collect();
    let (labels, _proba) = fit_complement::<f32>(&case);
    assert_eq!(labels, predict_ref, "ComplementNB f32 exact predict labels (HARD gate)");
}

/// HARD GATE: predict labels match sklearn EXACTLY, f64 (cpu; rocm skips).
#[test]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("complement_nb_f64_seed42.npz")).expect("load complement_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case.expect_f64("predict").iter().map(|&v| v.round() as i32).collect();
    let (labels, _proba) = fit_complement::<f64>(&case);
    assert_eq!(labels, predict_ref, "ComplementNB f64 exact predict labels (HARD gate)");
}

/// proba band + rows-sum-to-1, f64 (cpu; rocm skips).
#[test]
fn proba_band() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb proba f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("complement_nb_f64_seed42.npz")).expect("load complement_nb_f64");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_complement::<f64>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "ComplementNB f64 predict_proba");
}

/// proba band + rows-sum-to-1, f32.
#[test]
fn proba_band_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("complement_nb_f32_seed42.npz")).expect("load complement_nb_f32");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_complement::<f32>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F32, "ComplementNB f32 predict_proba");
}

/// D-02 litmus: bare builder().build() reproduces sklearn's default (norm=false).
#[test]
fn default_matches_sklearn() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb default f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("complement_nb_f64_seed42.npz")).expect("load complement_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case.expect_f64("predict").iter().map(|&v| v.round() as i32).collect();
    let proba_ref = case.expect_f64("predict_proba");
    let (labels, proba) = fit_complement::<f64>(&case);
    assert_eq!(labels, predict_ref, "default ComplementNB predict labels match sklearn");
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "default ComplementNB predict_proba");
}

/// norm=true: the second L1 normalization makes each weight row sum to 1.0
/// (feature_log_prob_ = logged / Σ_j logged), and the resulting predict_proba is
/// still a valid distribution (rows sum to 1.0). This exercises the norm code
/// path (which differs from the default −logged weights, Pitfall 6).
#[test]
fn norm_true() {
    if capability::skip_f64_with_log() {
        let backend = capability::active_backend_name();
        println!("complement_nb norm_true f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("complement_nb_f64_seed42.npz")).expect("load complement_nb_f64");
    assert_fixture_shape(&case);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = case.expect_f64("X").to_vec();
    let y_host: Vec<f64> = case.expect_f64("y").to_vec();
    let xq_host: Vec<f64> = case.expect_f64("Xq").to_vec();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);
    let xq_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xq_host);

    let clf = ComplementNB::<f64>::builder()
        .norm(true)
        .build::<f64>()
        .expect("norm=true ComplementNB builds");
    let clf = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("fit with norm=true");

    // L1-normalized weight rows each sum to 1.0 (the norm path invariant).
    let flp = clf.feature_log_prob(&pool).expect("fitted feature_log_prob");
    assert_eq!(flp.len(), N_CLASSES * N_FEATURES);
    for (c, row) in flp.chunks(N_FEATURES).enumerate() {
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() <= 1e-9,
            "norm=true weight row {c} sums to {s} (expected 1.0 — L1 normalized)"
        );
    }

    let proba: Vec<f64> =
        TypestatePredictProba::predict_proba(&clf, &mut pool, &xq_dev, (N_QUERY, N_FEATURES))
            .expect("predict_proba (norm=true)")
            .to_host(&pool);
    assert_rows_sum_to_one(&proba);
}

/// build()-rejection: alpha < 0 → BuildError::InvalidAlpha (D-05).
#[test]
fn build_rejects_bad_alpha() {
    let bad = ComplementNB::<f64>::builder().alpha(-1.0).build::<f64>().err();
    assert!(
        matches!(bad, Some(BuildError::InvalidAlpha { alpha, .. }) if alpha == -1.0),
        "alpha < 0 must be BuildError::InvalidAlpha, got {bad:?}"
    );
}

/// PoolStats no-leak gate (WR-07): live_bytes does not grow across a re-fit.
#[test]
fn refit_releases_buffers() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb refit f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("complement_nb_f64_seed42.npz")).expect("load complement_nb_f64");
    assert_fixture_shape(&case);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = case.expect_f64("X").to_vec();
    let y_host: Vec<f64> = case.expect_f64("y").to_vec();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    // Consuming-self Fit makes &mut self re-fit a type error; the gate becomes the
    // construct → fit (consuming) → drop(Fitted) cycle (umap_test fit_no_leak).
    let clf = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("default ComplementNB builds");
    let fitted = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    const REFITS: usize = 4;
    for k in 0..REFITS {
        let clf = ComplementNB::<f64>::builder()
            .build::<f64>()
            .expect("default ComplementNB builds");
        let fitted =
            TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
                .expect("re-fit");
        drop(fitted);
        let live = pool.stats().live_bytes;
        assert!(
            live <= live_after_first,
            "live_bytes grew across re-construct+fit {k}: {live} > first {live_after_first} (WR-07 leak)"
        );
    }
}
