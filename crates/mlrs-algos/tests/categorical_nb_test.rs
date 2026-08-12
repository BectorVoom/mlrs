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
//! The PERF-rewrite gates (row-major two-pass tabulation + the no-upload
//! host-slice fit arm) close the file:
//!
//!   - `parallel_passes_match_serial_reference` — the worker-chunked passes
//!     reproduce a naive serial tabulation BITWISE.
//!   - `worker_count_does_not_change_the_fit` — every `MLRS_CATNB_WORKERS`
//!     setting yields bitwise-identical tables.
//!   - `host_slice_fit_matches_device_fit` — the two fit entry points agree on
//!     every fitted table.
//!   - `fit_rejects_nonfinite_input` — NaN/±inf are rejected (the Python shim
//!     now relies on this instead of `check_array`'s own scan).
//!   - `rejection_reports_first_offender_regardless_of_chunking` — the error
//!     names the earliest offender in row-major order, not the worker's.
//!   - `host_slice_fit_guards_geometry` — the slice twin of `validate_geometry`.
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
use mlrs_backend::abflag;
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

    // A value that is still negative AFTER truncation is rejected; `-0.5` is
    // NOT, because it truncates to category 0 (see the fractional test below).
    let x_small: Vec<f64> = vec![0.0, 1.0, -0.5, 2.0];
    let x_small_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_small);
    let clf_s = CategoricalNB::<f64>::builder().build::<f64>().expect("builds");
    assert!(
        TypestateFit::fit(clf_s, &mut pool, &x_small_dev, Some(&y_dev), (2, 2)).is_ok(),
        "-0.5 truncates to category 0 and must be accepted (sklearn casts with dtype=\"int\")"
    );
}

/// A FRACTIONAL feature value is TRUNCATED toward zero, not rejected — and the
/// resulting model is identical to fitting the truncated values directly.
///
/// sklearn validates `CategoricalNB`'s X with `dtype="int"`, which casts
/// (numpy truncates toward zero) and only then calls `check_non_negative`. So
/// `1.7` is category 1, and `-0.5` is category 0. mlrs used to reject any
/// non-integer, which was stricter than sklearn on inputs sklearn accepts
/// silently — and made every estimator_check that fits this class on a
/// continuous fixture fail.
///
/// Fitting the truncated matrix directly must give the SAME model, which is
/// what pins the cast as truncation rather than rounding: with `round`, `1.7`
/// would land in category 2 and the two fits would diverge.
#[test]
fn fractional_input_truncates_like_sklearn() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let y_host: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0];
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y_host);

    //           1.7->1  0.2->0 | 2.9->2  1.1->1 | -0.5->0  2.8->2 | 0.4->0  1.9->1
    let frac: Vec<f64> = vec![1.7, 0.2, 2.9, 1.1, -0.5, 2.8, 0.4, 1.9];
    let trunc: Vec<f64> = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0];

    let fit_one = |x: &[f64], pool: &mut BufferPool<ActiveRuntime>| {
        let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, x);
        let clf = CategoricalNB::<f64>::builder().build::<f64>().expect("builds");
        let f = TypestateFit::fit(clf, pool, &xd, Some(&y_dev), (4, 2))
            .expect("a fractional categorical matrix is accepted (truncated)");
        let cats = f.n_categories().expect("fitted").to_vec();
        let flp = f.feature_log_prob().expect("fitted").to_vec();
        (cats, flp)
    };

    let (cat_f, flp_f) = fit_one(&frac, &mut pool);
    let (cat_t, flp_t) = fit_one(&trunc, &mut pool);
    assert_eq!(
        cat_f, cat_t,
        "the fractional fit must see the same category counts as the truncated one"
    );
    assert_eq!(
        flp_f, flp_t,
        "truncating in the caller and letting the fit truncate must give the SAME model"
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

// ===========================================================================
// PERF-rewrite regression gates (row-major two-pass tabulation + the no-upload
// host-slice fit arm). These lock the properties the rewrite could plausibly
// break: the two fit entry points must agree, the WORKER-CHUNKED passes must
// reproduce a naive serial tabulation, and the rejection messages must not
// depend on how the rows were split across workers.
// ===========================================================================

/// A deterministic categorical matrix + labels, large enough (`n·d` well past
/// the `PAR_MIN_ELEMS` threshold) that both fit passes run CHUNKED across the
/// scoped worker pool. Category counts differ per feature so the ragged
/// `n_categories_` / flat-offset indexing is genuinely exercised.
fn par_dataset() -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    const N: usize = 5_000;
    const D: usize = 13;
    const C: usize = 4;
    let mut x = Vec::with_capacity(N * D);
    let mut y = Vec::with_capacity(N);
    // A cheap reproducible LCG — no dev-dependency on an RNG crate.
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |m: u64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) % m
    };
    for _ in 0..N {
        for j in 0..D {
            // Feature j spans j+2 categories -> a genuinely ragged table.
            x.push(next((j + 2) as u64) as f64);
        }
        y.push(next(C as u64) as f64);
    }
    (x, y, N, D, C)
}

/// Every worker count produces BITWISE identical fitted tables.
///
/// This is the gate the `MLRS_CATNB_WORKERS` knob exists for: both passes split
/// the rows across a scoped pool, so a reduction that dropped a chunk, mis-sized
/// the last (short) chunk, or lost a per-worker count table would show up here
/// and nowhere else. `1` pins the fully serial arm; the counts are exact
/// integers accumulated in `f64`, so "close enough" is not the contract —
/// equality is.
#[test]
fn worker_count_does_not_change_the_fit() {
    let (x, y, n, d, _c) = par_dataset();

    let reference = {
        let _g = abflag::force("MLRS_CATNB_WORKERS", "1");
        CategoricalNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&x, &y, (n, d), None)
            .expect("serial fit")
    };

    for workers in ["2", "3", "5", "8", "64"] {
        let _g = abflag::force("MLRS_CATNB_WORKERS", workers);
        let got = CategoricalNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&x, &y, (n, d), None)
            .expect("chunked fit");
        assert_eq!(
            got.n_categories(),
            reference.n_categories(),
            "n_categories_ changed at {workers} workers (pass-1 max reduction)"
        );
        assert_eq!(
            got.feature_log_prob(),
            reference.feature_log_prob(),
            "feature_log_prob_ changed at {workers} workers (pass-2 table reduction)"
        );
        assert_eq!(
            got.class_count(),
            reference.class_count(),
            "class_count_ changed at {workers} workers"
        );
    }
}

/// The chunked two-pass tabulation reproduces a NAIVE serial reference for both
/// `n_categories_` (pass 1's per-feature max reduction) and `feature_log_prob_`
/// (pass 2's flat per-worker count tables summed back together).
#[test]
fn parallel_passes_match_serial_reference() {
    let (x, y, n, d, n_classes) = par_dataset();
    let alpha = 1.0f64;

    let clf = CategoricalNB::<f64>::builder().build::<f64>().expect("builds");
    let fitted = clf
        .fit_from_host_slice(&x, &y, (n, d), None)
        .expect("chunked fit succeeds");

    // --- Naive reference: per-feature max, then one count table per feature. ---
    let mut n_cat_ref = vec![0usize; d];
    for i in 0..n {
        for j in 0..d {
            n_cat_ref[j] = n_cat_ref[j].max(x[i * d + j] as usize + 1);
        }
    }
    assert_eq!(
        fitted.n_categories().expect("fitted"),
        n_cat_ref.as_slice(),
        "pass-1 per-feature max reduction diverged from the serial reference"
    );

    let mut class_count = vec![0.0f64; n_classes];
    for i in 0..n {
        class_count[y[i] as usize] += 1.0;
    }
    for j in 0..d {
        let flp_j = fitted.feature_log_prob_block(j).expect("fitted");
        let n_cat_j = n_cat_ref[j];
        let mut count = vec![0.0f64; n_classes * n_cat_j];
        for i in 0..n {
            count[y[i] as usize * n_cat_j + x[i * d + j] as usize] += 1.0;
        }
        for c in 0..n_classes {
            let denom = class_count[c] + alpha * n_cat_j as f64;
            for k in 0..n_cat_j {
                let want = ((count[c * n_cat_j + k] + alpha) / denom).ln();
                let got = flp_j[c * n_cat_j + k];
                assert_eq!(
                    got, want,
                    "feature_log_prob_[{j}][{c},{k}] diverged from the serial reference \
                     (counts are exact integers in f64, so this must be BITWISE equal)"
                );
            }
        }
    }
}

/// The no-upload [`CategoricalNB::fit_from_host_slice`] arm and the `DeviceArray`
/// [`TypestateFit::fit`] arm run the SAME body, so every fitted table must be
/// BITWISE identical. This is the gate that keeps the PyO3 host-slice route
/// honest against the typestate route the Rust callers use.
#[test]
fn host_slice_fit_matches_device_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb host-slice f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let via_device = TypestateFit::fit(
        CategoricalNB::<f64>::builder().build::<f64>().expect("builds"),
        &mut pool,
        &x_dev,
        Some(&y_dev),
        (n, d),
    )
    .expect("device fit");
    let via_host = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&x, &y, (n, d), None)
        .expect("host-slice fit");

    assert_eq!(via_host.classes(), via_device.classes(), "classes_ diverged");
    assert_eq!(
        via_host.n_categories(),
        via_device.n_categories(),
        "n_categories_ diverged"
    );
    assert_eq!(
        via_host.class_count(),
        via_device.class_count(),
        "class_count_ diverged"
    );
    assert_eq!(
        via_host.class_log_prior(),
        via_device.class_log_prior(),
        "class_log_prior_ diverged"
    );
    assert_eq!(
        via_host.feature_log_prob(),
        via_device.feature_log_prob(),
        "feature_log_prob_ diverged between the host-slice and device fit arms"
    );
}

/// A NaN feature value is REJECTED. Before the rewrite `(round(v) - v).abs() >
/// tol` was FALSE for NaN (every NaN comparison is false), so a NaN silently
/// rounded to category `0`; the test-visible consequence was masked only by the
/// Python shim's `check_array` scan. The scan now lives in this pass, so the
/// rejection has to be real here.
#[test]
fn fit_rejects_nonfinite_input() {
    for (label, bad) in [("NaN", f64::NAN), ("+inf", f64::INFINITY), ("-inf", f64::NEG_INFINITY)] {
        let y: Vec<f64> = vec![0.0, 1.0];
        let mut x: Vec<f64> = vec![0.0, 1.0, 1.0, 2.0];
        x[3] = bad;
        let got = CategoricalNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&x, &y, (2, 2), None)
            .err();
        assert!(
            matches!(got, Some(AlgoError::InvalidCategoricalInput { .. })),
            "a {label} feature value must be InvalidCategoricalInput, got {got:?}"
        );
    }
}

/// The reported offender is the FIRST one in ROW-MAJOR order, not whichever
/// worker happened to finish first. Pass 1 is split over row chunks, so without
/// the flat-index reduction the message would depend on the machine's core
/// count — a genuinely irreproducible error.
#[test]
fn rejection_reports_first_offender_regardless_of_chunking() {
    let (mut x, y, n, d, _c) = par_dataset();
    // Two invalid values, deliberately far apart so they land in DIFFERENT row
    // chunks on any plausible worker count.
    let early = 7 * d + 1;
    let late = (n - 5) * d + 2;
    x[early] = -3.0;
    x[late] = 0.25;

    let got = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&x, &y, (n, d), None)
        .err();
    match got {
        Some(AlgoError::InvalidCategoricalInput { reason, .. }) => assert!(
            reason.contains("-3"),
            "must report the EARLIEST offender (-3 at flat index {early}), got: {reason}"
        ),
        other => panic!("expected InvalidCategoricalInput, got {other:?}"),
    }
}

/// The host-slice arm carries the slice twin of the `validate_geometry` guard:
/// a length that does not match `n_samples · n_features`, an empty geometry, or
/// a mismatched `y` is a `ShapeMismatch`, never an out-of-bounds index.
#[test]
fn host_slice_fit_guards_geometry() {
    let x: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];
    let y: Vec<f64> = vec![0.0, 1.0];
    let build = || CategoricalNB::<f64>::builder().build::<f64>().expect("builds");

    for (label, xs, ys, shape) in [
        ("x too short", &x[..3], &y[..], (2usize, 2usize)),
        ("y too short", &x[..], &y[..1], (2, 2)),
        ("zero rows", &x[..0], &y[..0], (0, 2)),
        ("zero features", &x[..0], &y[..], (2, 0)),
    ] {
        let got = build().fit_from_host_slice(xs, ys, shape, None).err();
        assert!(
            matches!(got, Some(AlgoError::Prim(_))),
            "{label} must be a geometry PrimError, got {got:?}"
        );
    }
}

// ===========================================================================
// sample_weight (the fit parameter sklearn's `fit(X, y, sample_weight=None)`
// carries). The contract is stated by sklearn's own
// `check_sample_weight_equivalence_on_dense_data`: an INTEGER weight must be
// indistinguishable from repeating that row that many times, and a zero weight
// from dropping it. That is what these gates check, plus the rejections.
// ===========================================================================

/// Repeat row `i` of `(x, y)` `w[i]` times — the reference a weighted fit must
/// reproduce.
fn repeat_rows(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    d: usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    let mut xr = Vec::new();
    let mut yr = Vec::new();
    for (i, &wi) in w.iter().enumerate() {
        for _ in 0..(wi as usize) {
            xr.extend_from_slice(&x[i * d..(i + 1) * d]);
            yr.push(y[i]);
        }
    }
    let n = yr.len();
    (xr, yr, n)
}

/// Deterministic integer weights (including ZEROS, so the drop-a-row case is
/// covered) over [`par_dataset`], cycling `0,1,2,3` so every class sees each.
fn int_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 4) as f64).collect()
}

/// An integer-weighted fit equals the fit on the sample-REPEATED design, and a
/// zero weight drops its row. This is sklearn's own sample-weight contract.
#[test]
fn weighted_fit_equals_repeated_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb sample_weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let w = int_weights(n);
    let (xr, yr, nr) = repeat_rows(&x, &y, &w, d);
    
    let weighted = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&x, &y, (n, d), Some(&w))
        .expect("weighted fit");
    let repeated = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&xr, &yr, (nr, d), None)
        .expect("repeated fit");
    assert_eq!(weighted.classes(), repeated.classes(), "classes_ diverged");
    assert_eq!(
        weighted.n_categories(),
        repeated.n_categories(),
        "n_categories_ diverged"
    );
    // The fitted table is ONE flat buffer; walk it per feature so a divergence
    // is reported against the feature it belongs to rather than a flat index.
    for j in 0..weighted.n_categories().expect("fitted").len() {
        let a = weighted.feature_log_prob_block(j).expect("fitted");
        let b = repeated.feature_log_prob_block(j).expect("fitted");
        assert_band(a, b, 1e-12, &format!("feature_log_prob_[{j}] weighted vs repeated"));
    }
}

/// An all-ones `sample_weight` is the unweighted fit. Guards the weighted arm
/// against an off-by-one in the per-worker weight slicing, which a
/// uniform-weight fit would otherwise hide.
#[test]
fn all_ones_weight_equals_unweighted() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("categorical_nb ones-weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let ones = vec![1.0f64; n];
    
    let weighted = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&x, &y, (n, d), Some(&ones))
        .expect("ones-weighted fit");
    let plain = CategoricalNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&x, &y, (n, d), None)
        .expect("unweighted fit");
    assert_eq!(weighted.classes(), plain.classes(), "classes_ diverged");
    // The fitted table is ONE flat buffer; walk it per feature so a divergence
    // is reported against the feature it belongs to rather than a flat index.
    for j in 0..weighted.n_categories().expect("fitted").len() {
        let a = weighted.feature_log_prob_block(j).expect("fitted");
        let b = plain.feature_log_prob_block(j).expect("fitted");
        assert_band(a, b, 1e-12, &format!("feature_log_prob_[{j}] ones vs unweighted"));
    }
}

/// The three rejections sklearn's `_check_sample_weight` performs: a length
/// mismatch (which is also how a 2-D `sample_weight` arrives, ravelled, from the
/// Python shim), a non-finite or negative entry, and an ALL-ZERO vector —
/// the last carrying a message that mentions both "weight" and "zero", which is
/// what `check_all_zero_sample_weights_error` greps for.
#[test]
fn fit_rejects_bad_sample_weight() {
    let (x, y, n, d, _c) = par_dataset();
    
    let build = || CategoricalNB::<f64>::builder().build::<f64>().expect("builds");

    let short = vec![1.0f64; n - 1];
    assert!(
        matches!(
            build().fit_from_host_slice(&x, &y, (n, d), Some(&short)).err(),
            Some(AlgoError::Prim(_))
        ),
        "a length-mismatched sample_weight must be a geometry PrimError"
    );

    for (label, bad) in [("NaN", f64::NAN), ("+inf", f64::INFINITY), ("negative", -1.0)] {
        let mut w = vec![1.0f64; n];
        w[3] = bad;
        assert!(
            matches!(
                build().fit_from_host_slice(&x, &y, (n, d), Some(&w)).err(),
                Some(AlgoError::InvalidSampleWeight { index: 3, .. })
            ),
            "a {label} sample_weight must be InvalidSampleWeight at index 3"
        );
    }

    let zeros = vec![0.0f64; n];
    let err = build().fit_from_host_slice(&x, &y, (n, d), Some(&zeros)).err();
    assert!(
        matches!(err, Some(AlgoError::ZeroSampleWeightSum { .. })),
        "an all-zero sample_weight must be ZeroSampleWeightSum, got {err:?}"
    );
    let msg = format!("{}", err.expect("rejected"));
    assert!(
        msg.contains("weight") && msg.contains("zero"),
        "the all-zero message must mention both 'weight' and 'zero' \
         (check_all_zero_sample_weights_error greps for it): {msg}"
    );
}
