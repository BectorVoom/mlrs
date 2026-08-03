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
//! The PERF-rewrite gates (ONE sweep for both sufficient statistics + the
//! derived `epsilon_` + the no-upload host-slice fit arm) close the file:
//!
//!   - `worker_count_does_not_change_the_fit` — every `MLRS_GNB_WORKERS`
//!     setting yields identical fitted tables.
//!   - `parallel_sweep_matches_serial_reference` — the worker-chunked sweep
//!     reproduces a naive serial theta_/var_/epsilon_ reference.
//!   - `host_slice_fit_matches_device_fit` — the two fit entry points agree.
//!   - `fit_rejects_nonfinite_input` — NaN/±inf are rejected (the Python shim
//!     now relies on this instead of `check_array`'s own scan) while a NEGATIVE
//!     feature is still accepted (GaussianNB models real values).
//!   - `rejection_reports_first_offender_regardless_of_chunking` — the error
//!     names the earliest offender in row-major order, not the worker's.
//!   - `host_slice_fit_guards_geometry` — the slice twin of `validate_geometry`.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::naive_bayes::GaussianNB;
// Phase 16 (D-02): GaussianNB migrated to the typestate surface — consuming-self
// `Fit` and the `Fitted`-gated `PredictLabels`/`PredictProba` accessors are
// consumed via UFCS through these aliases.
use mlrs_algos::typestate::{
    Fit as TypestateFit, PredictLabels as TypestatePredictLabels,
    PredictProba as TypestatePredictProba,
};
use mlrs_backend::abflag;
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

    let clf = GaussianNB::<F>::builder()
        .build::<F>()
        .expect("default GaussianNB builds");
    let clf = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("GaussianNB::fit on a valid shape");

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

/// PoolStats no-leak gate (WR-07): live_bytes does not grow across a
/// re-CONSTRUCT + re-fit at the same shape. The consuming-self typestate `Fit`
/// makes a `&mut self` re-fit a type error, so the gate becomes the
/// born-with-convention "build a fresh `Unfit`, fit (consuming it), drop the
/// `Fitted` value" cycle (the umap_test `fit_no_leak` precedent): the dropped
/// `Fitted` returns its `theta_`/`var_` device buffers to the pool free-list,
/// which the next construct+fit reuses — so `live_bytes` stays flat.
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

    // Warm up: first construct+fit allocates theta_/var_; drop the Fitted value
    // (returns its buffers to the free-list) and record the steady live_bytes.
    let clf = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("default GaussianNB builds");
    let fitted = TypestateFit::fit(clf, &mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    // Re-CONSTRUCT + re-fit several times at the SAME shape; live_bytes must not
    // climb (the dropped theta_/var_ are released into the free-list and reused).
    const REFITS: usize = 4;
    for k in 0..REFITS {
        let clf = GaussianNB::<f64>::builder()
            .build::<f64>()
            .expect("default GaussianNB builds");
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
// PERF-rewrite regression gates (ONE sweep for Σx AND Σx², the epsilon_ column
// variances derived from those totals, and the no-upload host-slice fit arm).
// ===========================================================================

/// A deterministic REAL-valued design matrix + labels, large enough (`n·d` well
/// past `PAR_MIN_ELEMS`) that the sweep runs CHUNKED across the scoped worker
/// pool. Values straddle zero: GaussianNB models real features, and a negative
/// one must survive the sweep's finite-only check.
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
        for _ in 0..D {
            x.push(next(2001) as f64 / 100.0 - 10.0);
        }
        y.push(next(C as u64) as f64);
    }
    (x, y, N, D, C)
}

/// Host-materialize everything the sweep feeds: `theta_` and `var_` come from
/// the per-class Σx / Σx², and `epsilon_` from the whole-column totals the same
/// sweep produced, so a divergence anywhere in it lands in one of these four.
fn fitted_tables(
    est: mlrs_algos::naive_bayes::GaussianNB<f64, mlrs_algos::typestate::Fitted>,
    pool: &BufferPool<ActiveRuntime>,
) -> (Vec<i64>, Vec<f64>, Vec<f64>, f64) {
    let classes = est.classes().to_vec();
    let theta = est.theta(pool).expect("fitted");
    let var = est.var(pool).expect("fitted");
    let eps = est.epsilon().expect("fitted");
    (classes, theta, var, eps)
}

/// Every worker count produces identical fitted tables.
///
/// This is the gate the `MLRS_GNB_WORKERS` knob exists for: the sweep splits the
/// rows across a scoped pool, so a reduction that dropped a chunk, mis-sized the
/// last (short) chunk, or lost a per-worker table would show up here and nowhere
/// else. `1` pins the fully serial arm. The per-class sums are FLOATING point
/// here (real-valued features), so a different chunking reassociates them —
/// hence a tight band rather than bitwise equality.
#[test]
fn worker_count_does_not_change_the_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb workers f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let reference = {
        let _g = abflag::force("MLRS_GNB_WORKERS", "1");
        let est = GaussianNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
            .expect("serial fit");
        fitted_tables(est, &pool)
    };

    for workers in ["2", "3", "5", "8", "64"] {
        let got = {
            let _g = abflag::force("MLRS_GNB_WORKERS", workers);
            let est = GaussianNB::<f64>::builder()
                .build::<f64>()
                .expect("builds")
                .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
                .expect("chunked fit");
            fitted_tables(est, &pool)
        };
        assert_eq!(got.0, reference.0, "classes_ changed at {workers} workers");
        assert_band(&got.1, &reference.1, 1e-12, &format!("theta_ at {workers} workers"));
        assert_band(&got.2, &reference.2, 1e-12, &format!("var_ at {workers} workers"));
        assert_band(
            &[got.3],
            &[reference.3],
            1e-12,
            &format!("epsilon_ at {workers} workers"),
        );
    }
}

/// The chunked sweep reproduces a NAIVE serial reference for `theta_`, `var_`
/// AND `epsilon_`. `epsilon_` is the load-bearing one: it used to come from its
/// own COLUMN-strided pass over the design matrix and now falls out of the
/// per-class totals summed over `c`, so this is what proves the two are the same
/// quantity.
#[test]
fn parallel_sweep_matches_serial_reference() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb serial-ref f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, n_classes) = par_dataset();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("chunked fit succeeds");
    let (_classes, theta, var, eps) = fitted_tables(fitted, &pool);

    // --- Naive serial reference, straight from the definition. ---
    let mut class_count = vec![0.0f64; n_classes];
    let mut sums = vec![0.0f64; n_classes * d];
    let mut sumsqs = vec![0.0f64; n_classes * d];
    for i in 0..n {
        let c = y[i] as usize;
        class_count[c] += 1.0;
        for j in 0..d {
            let v = x[i * d + j];
            sums[c * d + j] += v;
            sumsqs[c * d + j] += v * v;
        }
    }
    // epsilon_ from a COLUMN pass over the raw matrix — the shape the fit no
    // longer runs, which is exactly why it is the right reference here.
    let mut max_col_var = 0.0f64;
    for j in 0..d {
        let (mut s, mut ss) = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = x[i * d + j];
            s += v;
            ss += v * v;
        }
        let mean = s / n as f64;
        max_col_var = max_col_var.max((ss / n as f64 - mean * mean).max(0.0));
    }
    let eps_ref = (1e-9 * max_col_var).max(f64::MIN_POSITIVE);
    assert_band(&[eps], &[eps_ref], 1e-10, "epsilon_ vs the column-pass reference");

    let mut theta_ref = vec![0.0f64; n_classes * d];
    let mut var_ref = vec![0.0f64; n_classes * d];
    for c in 0..n_classes {
        let n_c = class_count[c];
        for j in 0..d {
            let mean = sums[c * d + j] / n_c;
            theta_ref[c * d + j] = mean;
            var_ref[c * d + j] = (sumsqs[c * d + j] / n_c - mean * mean).max(0.0) + eps_ref;
        }
    }
    assert_band(&theta, &theta_ref, 1e-12, "theta_ vs the serial reference");
    assert_band(&var, &var_ref, 1e-10, "var_ vs the serial reference");
}

/// The no-upload host-slice arm and the `DeviceArray` `Fit::fit` arm run the
/// SAME body, so every fitted table must be BITWISE identical (same operand
/// values, same chunking, same order).
#[test]
fn host_slice_fit_matches_device_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb host-slice f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let via_device = TypestateFit::fit(
        GaussianNB::<f64>::builder().build::<f64>().expect("builds"),
        &mut pool,
        &x_dev,
        Some(&y_dev),
        (n, d),
    )
    .expect("device fit");
    let device_tables = fitted_tables(via_device, &pool);

    let via_host = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("host-slice fit");
    let host_tables = fitted_tables(via_host, &pool);

    assert_eq!(host_tables.0, device_tables.0, "classes_ diverged");
    assert_eq!(host_tables.1, device_tables.1, "theta_ diverged");
    assert_eq!(host_tables.2, device_tables.2, "var_ diverged");
    assert_eq!(host_tables.3, device_tables.3, "epsilon_ diverged");
}

/// A non-finite feature value is REJECTED by the fit's own sweep (the Python
/// shim passes `ensure_all_finite=False` and relies on this), while a NEGATIVE
/// value is ACCEPTED — GaussianNB models real-valued features, unlike the
/// count-based discrete variants whose sweep also rejects negatives.
#[test]
fn fit_rejects_nonfinite_input() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    for (label, bad) in [
        ("NaN", f64::NAN),
        ("+inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
    ] {
        let y: Vec<f64> = vec![0.0, 1.0];
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        x[3] = bad;
        let got = GaussianNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, (2, 2), None)
            .err();
        assert!(
            matches!(got, Some(AlgoError::InvalidLabels { .. })),
            "a {label} feature value must be rejected, got {got:?}"
        );
    }
    // The negative-value arm: a discrete-variant sweep would reject this.
    let y: Vec<f64> = vec![0.0, 1.0];
    let x: Vec<f64> = vec![-1.0, 2.0, -3.0, 4.0];
    assert!(
        GaussianNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, (2, 2), None)
            .is_ok(),
        "a NEGATIVE feature value must be accepted by GaussianNB"
    );
}

/// The reported offender is the FIRST one in ROW-MAJOR order, not whichever
/// worker happened to finish first. The sweep is split over row chunks, so
/// without the flat-index reduction the message would depend on the machine's
/// core count — a genuinely irreproducible error.
#[test]
fn rejection_reports_first_offender_regardless_of_chunking() {
    let (mut x, y, n, d, _c) = par_dataset();
    // Two invalid values, deliberately far apart so they land in DIFFERENT row
    // chunks on any plausible worker count.
    let early = 7 * d + 1;
    let late = (n - 5) * d + 2;
    x[early] = f64::NEG_INFINITY;
    x[late] = f64::NAN;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let got = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .err();
    match got {
        Some(AlgoError::InvalidLabels { reason, .. }) => assert!(
            reason.contains("-inf"),
            "must report the EARLIEST offender (-inf at flat index {early}), got: {reason}"
        ),
        other => panic!("expected InvalidLabels, got {other:?}"),
    }
}

/// The host-slice arm carries the slice twin of the `validate_geometry` guard:
/// a length that does not match `n_samples · n_features`, an empty geometry, or
/// a mismatched `y` is a `ShapeMismatch`, never an out-of-bounds index.
#[test]
fn host_slice_fit_guards_geometry() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = vec![0.0, 1.0];
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for (label, xs, ys, shape) in [
        ("x too short", &x[..3], &y[..], (2usize, 2usize)),
        ("y too short", &x[..], &y[..1], (2, 2)),
        ("zero rows", &x[..0], &y[..0], (0, 2)),
        ("zero features", &x[..0], &y[..], (2, 0)),
    ] {
        let got = GaussianNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, xs, ys, shape, None)
            .err();
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
        println!("gaussian_nb sample_weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let w = int_weights(n);
    let (xr, yr, nr) = repeat_rows(&x, &y, &w, d);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let weighted = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&w))
        .expect("weighted fit");
    let repeated = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &xr, &yr, (nr, d), None)
        .expect("repeated fit");
    let (wc, wtheta, wvar, weps) = fitted_tables(weighted, &pool);
    let (rc, rtheta, rvar, reps_) = fitted_tables(repeated, &pool);
    assert_eq!(wc, rc, "classes_ diverged");
    assert_band(&wtheta, &rtheta, 1e-12, "theta_ weighted vs repeated");
    // var_ carries epsilon_, which is the UNWEIGHTED max column variance —
    // repeating rows CHANGES that, so compare the variances with the two
    // epsilon_ floors taken back out and check epsilon_ itself separately.
    let wv: Vec<f64> = wvar.iter().map(|v| v - weps).collect();
    let rv: Vec<f64> = rvar.iter().map(|v| v - reps_).collect();
    assert_band(&wv, &rv, 1e-10, "var_ (epsilon_-free) weighted vs repeated");
}

/// An all-ones `sample_weight` is the unweighted fit. Guards the weighted arm
/// against an off-by-one in the per-worker weight slicing, which a
/// uniform-weight fit would otherwise hide.
#[test]
fn all_ones_weight_equals_unweighted() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb ones-weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let ones = vec![1.0f64; n];
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let weighted = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&ones))
        .expect("ones-weighted fit");
    let plain = GaussianNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("unweighted fit");
    let (wc, wtheta, wvar, weps) = fitted_tables(weighted, &pool);
    let (pc, ptheta, pvar, peps) = fitted_tables(plain, &pool);
    assert_eq!(wc, pc, "classes_ diverged");
    assert_band(&wtheta, &ptheta, 1e-12, "theta_ ones-weighted vs unweighted");
    assert_band(&wvar, &pvar, 1e-12, "var_ ones-weighted vs unweighted");
    assert_band(&[weps], &[peps], 1e-12, "epsilon_ ones-weighted vs unweighted");
}

/// The three rejections sklearn's `_check_sample_weight` performs: a length
/// mismatch (which is also how a 2-D `sample_weight` arrives, ravelled, from the
/// Python shim), a non-finite or negative entry, and an ALL-ZERO vector —
/// the last carrying a message that mentions both "weight" and "zero", which is
/// what `check_all_zero_sample_weights_error` greps for.
#[test]
fn fit_rejects_bad_sample_weight() {
    let (x, y, n, d, _c) = par_dataset();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let build = || GaussianNB::<f64>::builder().build::<f64>().expect("builds");

    let short = vec![1.0f64; n - 1];
    assert!(
        matches!(
            build().fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&short)).err(),
            Some(AlgoError::Prim(_))
        ),
        "a length-mismatched sample_weight must be a geometry PrimError"
    );

    for (label, bad) in [("NaN", f64::NAN), ("+inf", f64::INFINITY), ("negative", -1.0)] {
        let mut w = vec![1.0f64; n];
        w[3] = bad;
        assert!(
            matches!(
                build().fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&w)).err(),
                Some(AlgoError::InvalidSampleWeight { index: 3, .. })
            ),
            "a {label} sample_weight must be InvalidSampleWeight at index 3"
        );
    }

    let zeros = vec![0.0f64; n];
    let err = build().fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&zeros)).err();
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
