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
//! The PERF-rewrite gates (the fused single-sweep count + the no-upload
//! host-slice fit arm) close the file:
//!
//!   - `worker_count_does_not_change_the_fit` — every `MLRS_CNB_WORKERS` setting
//!     yields identical fitted tables.
//!   - `parallel_sweep_matches_serial_reference` — the worker-chunked sweep
//!     reproduces a naive serial reference.
//!   - `host_slice_fit_matches_device_fit` — the two fit entry points agree.
//!   - `fit_rejects_nonfinite_input` — NaN/±inf are rejected (the Python shim
//!     now relies on this instead of `check_array`'s own scan).
//!   - `rejection_reports_first_offender_regardless_of_chunking` — the error
//!     names the earliest offender in row-major order, not the worker's.
//!   - `host_slice_fit_guards_geometry` — the slice twin of `validate_geometry`.
//!
//! f64 cases carry the `skip_f64_with_log` gate (D-07). Per AGENTS.md §2 tests
//! live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::naive_bayes::ComplementNB;
// Phase 16 (D-02): ComplementNB migrated to the typestate surface — consuming-self
// `Fit` + `Fitted`-gated accessors consumed via UFCS through these aliases.
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

// ===========================================================================
// PERF-rewrite regression gates (the fused single-sweep count + the no-upload
// host-slice fit arm). These lock the properties the rewrite could plausibly
// break: the two fit entry points must agree, the WORKER-CHUNKED sweep must
// reproduce a naive serial reference, and the rejection messages must not
// depend on how the rows were split across workers.
// ===========================================================================

/// A deterministic design matrix + labels, large enough (`n·d` well past
/// `PAR_MIN_ELEMS`) that the sweep runs CHUNKED across the scoped worker pool.
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
            x.push(next(9) as f64);
        }
        y.push(next(C as u64) as f64);
    }
    (x, y, N, D, C)
}

/// Host-materialize the fitted tables a divergence in the count sweep shows up
/// in.
fn fitted_tables(
    est: mlrs_algos::naive_bayes::ComplementNB<f64, mlrs_algos::typestate::Fitted>,
    pool: &BufferPool<ActiveRuntime>,
) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let classes = est.classes().to_vec();
    let prior = est.class_log_prior().expect("fitted").to_vec();
    let flp = est.feature_log_prob(pool).expect("fitted");
    (classes, prior, flp)
}

/// Every worker count produces identical fitted tables.
///
/// This is the gate the `MLRS_CNB_WORKERS` knob exists for: the sweep splits the rows
/// across a scoped pool, so a reduction that dropped a chunk, mis-sized the last
/// (short) chunk, or lost a per-worker table would show up here and nowhere
/// else. `1` pins the fully serial arm.
#[test]
fn worker_count_does_not_change_the_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb workers f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let reference = {
        let _g = abflag::force("MLRS_CNB_WORKERS", "1");
        let est = ComplementNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
            .expect("serial fit");
        fitted_tables(est, &pool)
    };

    for workers in ["2", "3", "5", "8", "64"] {
        let got = {
            let _g = abflag::force("MLRS_CNB_WORKERS", workers);
            let est = ComplementNB::<f64>::builder()
                .build::<f64>()
                .expect("builds")
                .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
                .expect("chunked fit");
            fitted_tables(est, &pool)
        };
        assert_eq!(got.0, reference.0, "classes_ changed at {workers} workers");
        assert_eq!(
            got.1, reference.1,
            "class_log_prior_ changed at {workers} workers"
        );
        // The counts are exact integers here (the fixture values are integral),
        // so the derived table must be BITWISE equal across worker counts.
        assert_eq!(
            got.2, reference.2,
            "feature_log_prob_ changed at {workers} workers (count-table reduction)"
        );
    }
}

/// The chunked sweep reproduces a NAIVE serial per-class column sum.
#[test]
fn parallel_sweep_matches_serial_reference() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb serial-ref f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, n_classes) = par_dataset();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("chunked fit succeeds");
    let flp = fitted.feature_log_prob(&pool).expect("fitted");

    let mut count = vec![0.0f64; n_classes * d];
    for i in 0..n {
        let c = y[i] as usize;
        for j in 0..d {
            count[c * d + j] += x[i * d + j];
        }
    }
    let want = {
        let alpha = 1.0f64;
        let mut feature_all = vec![0.0f64; d];
        for c in 0..n_classes {
            for j in 0..d {
                feature_all[j] += count[c * d + j];
            }
        }
        let mut w = vec![0.0f64; n_classes * d];
        for c in 0..n_classes {
            let comp: Vec<f64> = (0..d)
                .map(|j| feature_all[j] + alpha - count[c * d + j])
                .collect();
            let comp_sum: f64 = comp.iter().sum();
            for j in 0..d {
                w[c * d + j] = -((comp[j] / comp_sum).ln());
            }
        }
        w
    };
    assert_eq!(
        flp.len(),
        want.len(),
        "feature_log_prob_ length diverged from the serial reference"
    );
    for (k, (&g, &w)) in flp.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "feature_log_prob_[{k}] diverged from the serial reference \
             (the per-class sums are exact integers, so this must be BITWISE equal)"
        );
    }
}

/// The no-upload host-slice arm and the `DeviceArray` `Fit::fit` arm run the
/// SAME body, so every fitted table must be identical.
#[test]
fn host_slice_fit_matches_device_fit() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb host-slice f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let via_device = TypestateFit::fit(
        ComplementNB::<f64>::builder().build::<f64>().expect("builds"),
        &mut pool,
        &x_dev,
        Some(&y_dev),
        (n, d),
    )
    .expect("device fit");
    let device_tables = fitted_tables(via_device, &pool);

    let via_host = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("host-slice fit");
    let host_tables = fitted_tables(via_host, &pool);

    assert_eq!(host_tables.0, device_tables.0, "classes_ diverged");
    assert_eq!(host_tables.1, device_tables.1, "class_log_prior_ diverged");
    assert_eq!(
        host_tables.2, device_tables.2,
        "feature_log_prob_ diverged between the host-slice and device fit arms"
    );
}

/// A non-finite feature value is REJECTED by the fit's own sweep. The Python
/// shim passes `ensure_all_finite=False` and relies on this instead of
/// `check_array`'s scan, so the rejection has to be real here.
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
        let got = ComplementNB::<f64>::builder()
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, (2, 2), None)
            .err();
        assert!(
            matches!(got, Some(AlgoError::InvalidFeatureInput { .. })),
            "a {label} feature value must be rejected, got {got:?}"
        );
    }
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
    x[early] = -3.0;
    x[late] = f64::INFINITY;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let got = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .err();
    match got {
        Some(AlgoError::InvalidFeatureInput { reason, .. }) => assert!(
            reason.contains("-3"),
            "must report the EARLIEST offender at flat index {early}, got: {reason}"
        ),
        other => panic!("expected InvalidFeatureInput, got {other:?}"),
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
        let got = ComplementNB::<f64>::builder()
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
        println!("complement_nb sample_weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let w = int_weights(n);
    let (xr, yr, nr) = repeat_rows(&x, &y, &w, d);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let weighted = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&w))
        .expect("weighted fit");
    let repeated = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &xr, &yr, (nr, d), None)
        .expect("repeated fit");
    let (wc, wprior, wflp) = fitted_tables(weighted, &pool);
    let (rc, rprior, rflp) = fitted_tables(repeated, &pool);
    assert_eq!(wc, rc, "classes_ diverged");
    assert_band(&wprior, &rprior, 1e-12, "class_log_prior_ weighted vs repeated");
    assert_band(&wflp, &rflp, 1e-12, "feature_log_prob_ weighted vs repeated");
}

/// An all-ones `sample_weight` is the unweighted fit. Guards the weighted arm
/// against an off-by-one in the per-worker weight slicing, which a
/// uniform-weight fit would otherwise hide.
#[test]
fn all_ones_weight_equals_unweighted() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("complement_nb ones-weight f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (x, y, n, d, _c) = par_dataset();
    let ones = vec![1.0f64; n];
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let weighted = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), Some(&ones))
        .expect("ones-weighted fit");
    let plain = ComplementNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, (n, d), None)
        .expect("unweighted fit");
    let (wc, wprior, wflp) = fitted_tables(weighted, &pool);
    let (pc, pprior, pflp) = fitted_tables(plain, &pool);
    assert_eq!(wc, pc, "classes_ diverged");
    assert_band(&wprior, &pprior, 1e-12, "class_log_prior_ ones vs unweighted");
    assert_band(&wflp, &pflp, 1e-12, "feature_log_prob_ ones vs unweighted");
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
    let build = || ComplementNB::<f64>::builder().build::<f64>().expect("builds");

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
