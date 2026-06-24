//! Plan 11-04 Wave-1 — CategoricalNB (NB-05) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. CategoricalNB fits a RAGGED
//! per-feature `feature_log_prob_` (`feature_log_prob_[j]` is the
//! `n_classes × n_categories_[j]` log-prob matrix, variable category count —
//! Pitfall 7) with `feature_log_prob_[j][c,k] = log((count[c,k]+alpha) /
//! (class_count[c] + alpha·n_categories_j))` (Pitfall 4 — the denominator
//! smoothing is alpha·n_categories_j), MinCategories padding (D-04), and
//! non-negative-integer input validation (T-11-04-01). Predict sums the
//! per-feature looked-up log-probs (lookup index guarded against n_categories_j,
//! T-11-04-02) + class_log_prior_, then `log_sum_exp_normalize` + argmax_decode:
//!
//!   - `exact_labels` / `exact_labels_f32` — `predict_labels(Xq)` match sklearn
//!     EXACTLY (the HARD gate, integer labels, no band).
//!   - `proba_band` — `predict_proba(Xq)` within band AND every row sums to 1.0.
//!   - `default_matches_sklearn` — bare `builder().build()` reproduces sklearn's
//!     default `CategoricalNB` (alpha=1.0, min_categories=None).
//!   - `min_categories` — `MinCategories::{Uniform,PerFeature}` padding yields the
//!     sklearn-matching predictions (padding-beyond-observed leaves labels/proba
//!     unchanged from the inferred-categories default).
//!   - `fit_rejects_bad_input` — negative / non-integer X → InvalidCategoricalInput.
//!   - `build_rejects_bad_alpha` — `build()` rejects `alpha < 0`.
//!   - `refit_releases_buffers` — the PoolStats no-leak gate across a re-fit.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate (cpu runs; rocm skips,
//! D-07). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::naive_bayes::{CategoricalNB, MinCategories};
// Phase 16 (D-02): CategoricalNB migrated to the typestate surface — consuming-
// self `Fit` + `Fitted`-gated accessors consumed via UFCS through these aliases.
use mlrs_algos::typestate::{
    Fit as TypestateFit, PredictLabels as TypestatePredictLabels,
    PredictProba as TypestatePredictProba, Unfit,
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

/// predict_proba bands. f64 is the global 1e-5 oracle gate; f32 at 1e-3 (the
/// categorical joint-LL is a host sum of looked-up log-probs — linear, so f32
/// round-off is well below the GaussianNB quadratic worst case, A4).
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
        _ => unreachable!("categorical_nb fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("categorical_nb fixtures are f32/f64 only"),
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

fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(case.expect_f64("X").len(), N_SAMPLES * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("predict").len(), N_QUERY);
    assert_eq!(case.expect_f64("predict_proba").len(), N_QUERY * N_CLASSES);
}

/// Build (with the given builder) + fit a `CategoricalNB` on the fixture and
/// return host `(predict_labels(Xq), predict_proba(Xq))`.
fn fit_categorical_with<F>(
    case: &OracleCase,
    clf: CategoricalNB<F, Unfit>,
) -> (Vec<i32>, Vec<f64>)
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

    let clf = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("CategoricalNB::fit on a valid shape");

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

/// The sklearn-default `CategoricalNB` (alpha=1.0, min_categories=None → Infer).
fn fit_categorical<F>(case: &OracleCase) -> (Vec<i32>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let clf = CategoricalNB::<F>::builder()
        .build::<F>()
        .expect("default CategoricalNB builds");
    fit_categorical_with(case, clf)
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

/// HARD GATE: predict labels match sklearn EXACTLY, f32.
#[test]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("categorical_nb_f32_seed42.npz")).expect("load categorical_nb_f32");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (labels, _proba) = fit_categorical::<f32>(&case);
    assert_eq!(labels, predict_ref, "CategoricalNB f32 exact predict labels (HARD gate)");
}

/// HARD GATE: predict labels match sklearn EXACTLY, f64 (cpu; rocm skips).
#[test]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("categorical_nb_f64_seed42.npz")).expect("load categorical_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (labels, _proba) = fit_categorical::<f64>(&case);
    assert_eq!(labels, predict_ref, "CategoricalNB f64 exact predict labels (HARD gate)");
}

/// proba band + rows-sum-to-1, f64 (cpu; rocm skips).
#[test]
fn proba_band() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb proba f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("categorical_nb_f64_seed42.npz")).expect("load categorical_nb_f64");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_categorical::<f64>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "CategoricalNB f64 predict_proba");
}

/// proba band + rows-sum-to-1, f32.
#[test]
fn proba_band_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("categorical_nb_f32_seed42.npz")).expect("load categorical_nb_f32");
    assert_fixture_shape(&case);
    let proba_ref = case.expect_f64("predict_proba");
    let (_labels, proba) = fit_categorical::<f32>(&case);
    assert_rows_sum_to_one(&proba);
    assert_band(&proba, proba_ref, PROBA_BAND_F32, "CategoricalNB f32 predict_proba");
}

/// D-02 litmus: bare `builder().build()` reproduces sklearn's default
/// (min_categories=Infer, alpha=1.0, fit_prior=true).
#[test]
fn default_matches_sklearn() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb default f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("categorical_nb_f64_seed42.npz")).expect("load categorical_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let proba_ref = case.expect_f64("predict_proba");
    let (labels, proba) = fit_categorical::<f64>(&case);
    assert_eq!(labels, predict_ref, "default CategoricalNB predict labels match sklearn");
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "default CategoricalNB predict_proba");
}

/// Per-variant: MinCategories::{Uniform,PerFeature} padding. The fixture's
/// per-feature observed-max gives n_categories_j = NB_N_CATEGORIES = 4. Padding
/// to a value <= 4 (Uniform(4), PerFeature([4,4,4,4])) leaves the fitted shape
/// and predictions IDENTICAL to the sklearn default (min_categories=None) — the
/// pad-only-grows contract `n_categories_j = max(observed+1, min_j)`. Padding
/// BEYOND the observed max grows each feature's category table with all-unseen
/// (count==0, smoothed) cells; those cells never appear in Xq (A3) so the labels
/// and the proba still match sklearn's default fit.
#[test]
fn min_categories() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb min_categories f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("categorical_nb_f64_seed42.npz")).expect("load categorical_nb_f64");
    assert_fixture_shape(&case);
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();

    // Uniform(4): every feature already has 4 observed categories → no-op pad,
    // identical to the default fit.
    let clf_u = CategoricalNB::<f64>::builder()
        .min_categories(MinCategories::Uniform(4))
        .build::<f64>()
        .expect("CategoricalNB Uniform(4) builds");
    let (labels_u, _proba_u) = fit_categorical_with(&case, clf_u);
    assert_eq!(
        labels_u, predict_ref,
        "MinCategories::Uniform(4) (== observed) matches the sklearn default labels"
    );

    // PerFeature([6,6,6,6]): pads each feature to 6 categories (2 all-unseen
    // smoothed cells per feature). Those categories never appear in Xq (A3), so
    // the labels are unchanged from the default fit.
    let clf_p = CategoricalNB::<f64>::builder()
        .min_categories(MinCategories::PerFeature(vec![6, 6, 6, 6]))
        .build::<f64>()
        .expect("CategoricalNB PerFeature builds");
    let (labels_p, proba_p) = fit_categorical_with(&case, clf_p);
    assert_rows_sum_to_one(&proba_p);
    assert_eq!(
        labels_p, predict_ref,
        "MinCategories::PerFeature padding-beyond-observed keeps the sklearn default labels (A3: no unseen at predict)"
    );
}

/// Per-variant: negative / non-integer categorical input →
/// AlgoError::InvalidCategoricalInput (T-11-04-01).
#[test]
fn fit_rejects_bad_input() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // A 2x2 categorical matrix with a NEGATIVE entry.
    let y_host: Vec<f64> = vec![0.0, 1.0];
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    let x_neg: Vec<f64> = vec![0.0, 1.0, -1.0, 2.0];
    let x_neg_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_neg);
    let clf = CategoricalNB::<f64>::builder().build::<f64>().expect("builds");
    let neg = TypestateFit::fit(clf, &mut pool, &x_neg_dev, Some(&y_dev), (2, 2)).err();
    assert!(
        matches!(neg, Some(AlgoError::InvalidCategoricalInput { .. })),
        "negative categorical value must be InvalidCategoricalInput, got {neg:?}"
    );

    // A non-INTEGER entry (0.5).
    let x_frac: Vec<f64> = vec![0.0, 1.0, 0.5, 2.0];
    let x_frac_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_frac);
    let clf2 = CategoricalNB::<f64>::builder().build::<f64>().expect("builds");
    let frac = TypestateFit::fit(clf2, &mut pool, &x_frac_dev, Some(&y_dev), (2, 2)).err();
    assert!(
        matches!(frac, Some(AlgoError::InvalidCategoricalInput { .. })),
        "non-integer categorical value must be InvalidCategoricalInput, got {frac:?}"
    );
}

/// build()-rejection: alpha < 0 → BuildError::InvalidAlpha (D-05).
#[test]
fn build_rejects_bad_alpha() {
    let bad = CategoricalNB::<f64>::builder().alpha(-1.0).build::<f64>().err();
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
        println!("categorical_nb refit f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("categorical_nb_f64_seed42.npz")).expect("load categorical_nb_f64");
    assert_fixture_shape(&case);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = case.expect_f64("X").to_vec();
    let y_host: Vec<f64> = case.expect_f64("y").to_vec();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    // Consuming-self Fit makes &mut self re-fit a type error; the gate becomes the
    // construct → fit (consuming) → drop(Fitted) cycle (umap_test fit_no_leak).
    // CategoricalNB holds NO device buffer (ragged host tables), so live_bytes is
    // trivially flat, but the gate is kept for cross-NB uniformity.
    let clf = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("default CategoricalNB builds");
    let fitted = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    const REFITS: usize = 4;
    for k in 0..REFITS {
        let clf = CategoricalNB::<f64>::builder()
            .build::<f64>()
            .expect("default CategoricalNB builds");
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
