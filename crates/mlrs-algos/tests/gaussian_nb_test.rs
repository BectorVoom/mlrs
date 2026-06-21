//! Plan 11-02 Wave-1 — GaussianNB (NB-01) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The estimator fits per-class
//! `theta_`/`var_` from the validated `class_grouped_sum`/`sumsq` GATHERs, floors
//! `var_` by the GLOBAL `epsilon_ = var_smoothing · max_j Var(X[:,j])` (Pitfall
//! 3), and predicts host-f64 joint LL normalized by `log_sum_exp_normalize`:
//!
//!   - `exact_labels` / `exact_labels_f32` — `predict_labels(Xq)` match sklearn
//!     EXACTLY (the HARD gate, integer labels, no band).
//!   - `proba_band` — `predict_proba(Xq)` value-match within the documented band
//!     AND every row sums to 1.0 ± 1e-6 (GaussianNB log-proba gets the WIDEST
//!     f32 band, A4).
//!   - `default_matches_sklearn` — bare `builder().build()` reproduces sklearn's
//!     default `GaussianNB` (var_smoothing=1e-9, priors=None): its
//!     predict/predict_proba equal the default-fixture references (D-02 litmus).
//!   - `build_rejects_bad_var_smoothing` — `build()` rejects `var_smoothing < 0`
//!     (D-05 validate-at-build).
//!   - `refit_releases_buffers` — the PoolStats no-leak gate across a re-fit.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::naive_bayes::GaussianNB;
use mlrs_algos::traits::{Fit, PredictLabels, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// GaussianNB fixture geometry (gen_oracle.py `NB_N_SAMPLES` // `NB_N_CLASSES` ×
/// `NB_N_FEATURES`, `NB_N_QUERY` // `NB_N_CLASSES` query rows, 3 classes).
const N_SAMPLES: usize = 39;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 6;
const N_CLASSES: usize = 3;

/// predict_proba bands. The f64 band is the global 1e-5 oracle gate (CLAUDE.md
/// correctness contract). The f32 band is set from the MEASURED f32-vs-f64
/// residual (A4 — GaussianNB's per-feature Gaussian LL is the widest of the five
/// NB variants because the quadratic `(x−θ)²/var` term amplifies f32 round-off
/// before the log-sum-exp): the observed max abs residual on the seed-42 fixture
/// is ~3e-4, so a 1e-3 band is the tight-but-non-flaky bound.
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
        _ => unreachable!("gaussian_nb fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("gaussian_nb fixtures are f32/f64 only"),
    }
}

fn assert_band(got: &[f64], expected: &[f64], band: f64, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= band + band * e.abs(),
            "{what}: band failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e} (band={band:e})"
        );
    }
}

/// Assert the fixture's array shapes match the pinned NB geometry.
fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(
        case.expect_f64("X").len(),
        N_SAMPLES * N_FEATURES,
        "X is N_SAMPLES x N_FEATURES"
    );
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES, "y is N_SAMPLES");
    assert_eq!(
        case.expect_f64("Xq").len(),
        N_QUERY * N_FEATURES,
        "Xq is N_QUERY x N_FEATURES"
    );
    assert_eq!(
        case.expect_f64("predict").len(),
        N_QUERY,
        "predict is N_QUERY labels"
    );
    assert_eq!(
        case.expect_f64("predict_proba").len(),
        N_QUERY * N_CLASSES,
        "predict_proba is N_QUERY x N_CLASSES"
    );
}

/// Build (sklearn defaults) + fit a `GaussianNB` on the fixture and return host
/// `(predict_labels(Xq), predict_proba(Xq))`.
fn fit_gaussian<F>(case: &OracleCase) -> (Vec<i32>, Vec<f64>)
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

    let mut clf = GaussianNB::<F>::builder()
        .build::<F>()
        .expect("default GaussianNB builds");
    clf.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("GaussianNB::fit on a valid shape");

    let labels = clf
        .predict_labels(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_labels after fit")
        .to_host(&pool);
    let proba: Vec<f64> = clf
        .predict_proba(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_proba after fit")
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();

    (labels, proba)
}

/// Assert every `predict_proba` row sums to 1.0 ± 1e-6 (host log-sum-exp).
fn assert_rows_sum_to_one(proba: &[f64]) {
    for (r, row) in proba.chunks(N_CLASSES).enumerate() {
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() <= 1e-6,
            "predict_proba row {r} sums to {s} (expected 1.0 ± 1e-6)"
        );
    }
}

/// HARD GATE: predict labels match sklearn EXACTLY (integers, no band), f32.
#[test]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("gaussian_nb_f32_seed42.npz")).expect("load gaussian_nb_f32");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (labels, _proba) = fit_gaussian::<f32>(&case);
    assert_eq!(
        labels, predict_ref,
        "GaussianNB f32 exact predict labels (HARD gate)"
    );
}

/// HARD GATE: predict labels match sklearn EXACTLY, f64 (cpu runs; rocm skips).
#[test]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (labels, _proba) = fit_gaussian::<f64>(&case);
    assert_eq!(
        labels, predict_ref,
        "GaussianNB f64 exact predict labels (HARD gate)"
    );
}

/// proba band + rows-sum-to-1: predict_proba value-match within the documented
/// band, f32 (the widest band per A4); every row normalizes to 1.0 ± 1e-6.
#[test]
fn proba_band_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("gaussian_nb_f32_seed42.npz")).expect("load gaussian_nb_f32");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_gaussian::<f32>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F32, "GaussianNB f32 predict_proba");
}

/// proba band + rows-sum-to-1: predict_proba value-match within band, f64
/// (cpu runs; rocm skips); every row normalizes to 1.0 ± 1e-6.
#[test]
fn proba_band() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb proba f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_gaussian::<f64>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "GaussianNB f64 predict_proba");
}

/// D-02 litmus: bare `builder().build()` (var_smoothing=1e-9, priors=None)
/// reproduces sklearn's default `GaussianNB` — its predict labels match sklearn
/// EXACTLY and its predict_proba matches within the f64 band (the default fixture
/// was generated from the sklearn-default constructor). cpu runs; rocm skips.
#[test]
fn default_matches_sklearn() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb default f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let proba_ref = case.expect_f64("predict_proba");
    // The default-constructor build is exactly what fit_gaussian uses (no setters).
    let (labels, proba) = fit_gaussian::<f64>(&case);
    assert_eq!(
        labels, predict_ref,
        "default GaussianNB predict labels match sklearn (D-02 litmus)"
    );
    assert_band(
        &proba,
        proba_ref,
        PROBA_BAND_F64,
        "default GaussianNB predict_proba matches sklearn (D-02 litmus)",
    );
}

/// build()-rejection: var_smoothing < 0 → BuildError::InvalidVarSmoothing (D-05).
#[test]
fn build_rejects_bad_var_smoothing() {
    let bad = GaussianNB::<f64>::builder()
        .var_smoothing(-1.0)
        .build::<f64>()
        .err();
    assert!(
        matches!(
            bad,
            Some(BuildError::InvalidVarSmoothing { var_smoothing, .. }) if var_smoothing == -1.0
        ),
        "var_smoothing < 0 must be BuildError::InvalidVarSmoothing, got {bad:?}"
    );
}

/// PoolStats no-leak gate (WR-07): live_bytes does not grow across a re-fit at
/// the same shape — the fit releases the prior `theta_`/`var_` device buffers
/// before storing the new ones (and the GATHER helpers release their scratch).
#[test]
fn refit_releases_buffers() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb refit f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = case.expect_f64("X").to_vec();
    let y_host: Vec<f64> = case.expect_f64("y").to_vec();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    let mut clf = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("default GaussianNB builds");

    // Warm up: first fit allocates theta_/var_; record the steady live_bytes.
    clf.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("first fit");
    let live_after_first = pool.stats().live_bytes;

    // Re-fit several times at the SAME shape; live_bytes must not climb (the old
    // theta_/var_ are released into the free-list and reused — WR-07).
    const REFITS: usize = 4;
    for k in 0..REFITS {
        clf.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
            .expect("re-fit");
        let live = pool.stats().live_bytes;
        assert!(
            live <= live_after_first,
            "live_bytes grew across re-fit {k}: {live} > first {live_after_first} (WR-07 leak)"
        );
    }
}
