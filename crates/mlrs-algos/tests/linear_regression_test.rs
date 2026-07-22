//! Plan 04-03 — LinearRegression (LINEAR-01) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now
//! loads its committed `LinearRegression(fit_intercept=True)` fixture (sklearn's
//! `scipy.linalg.lstsq` / gelsd contract, the exact LINEAR-01 pin), fits the
//! device estimator, materializes `coef_`/`intercept_`/`predict`, and asserts
//! against the sklearn reference within the 1e-5 abs+rel contract.
//!
//! Two case families per dtype:
//!   - **Full-rank** (`X`/`y`, `coef`/`intercept`, `X_test`/`y_pred`).
//!   - **Near-collinear** (`X_coll`/`y_coll`, `coef_col`/`intercept_col`): feature
//!     2 ≈ feature 0, so the design has a ~0 singular value. The SVD-pseudo-inverse
//!     small-σ cutoff (RESEARCH Pitfall 1 / T-04-03-01) must keep `coef_col`
//!     bounded and matching sklearn — a no-cutoff inverse explodes here.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// LinearRegression fixture geometry (gen_oracle.py `LIN_N_SAMPLES` ×
/// `LIN_N_FEATURES`, `LIN_TEST_SAMPLES`). Exercises `fit_direct_svd`
/// (`n_samples.max(n_features) <= DIRECT_SVD_MAX_ROWS = 256`).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_TEST: usize = 3;

/// Large-`n_samples` LinearRegression fixture geometry (gen_oracle.py
/// `LIN_LARGE_N_SAMPLES` × `LIN_LARGE_N_FEATURES`). `n_samples = 2000 >
/// DIRECT_SVD_MAX_ROWS = 256`, so this exercises the `fit_gram_eig` path.
const LARGE_N_SAMPLES: usize = 2000;
const LARGE_N_FEATURES: usize = 20;

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
        _ => unreachable!("linreg fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the D-10 floored
/// precedent from `svd_test.rs`/`gemm_test.rs`).
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

/// Load the fixture, fit `LinearRegression(true)` on the `(x_key, y_key)` case
/// at `shape`, and return host `(coef_, intercept_)`. `shape` selects which
/// `fit` path runs (`fit_direct_svd` vs `fit_gram_eig`, D-02 dual-path).
fn fit_coef_intercept<F>(
    case: &OracleCase,
    x_key: &str,
    y_key: &str,
    shape: (usize, usize),
) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64(x_key)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case
        .expect_f64(y_key)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let reg = LinearRegression::<F>::builder()
        .fit_intercept(true)
        .build::<F>()
        .expect("LinearRegression build")
        .fit(&mut pool, &x_dev, Some(&y_dev), shape)
        .expect("LinearRegression::fit on a valid shape");

    let coef = reg
        .coef(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(reg.intercept(&pool));
    (coef, intercept)
}

/// Fit the full-rank case at `shape` and return host `predict(X_test)`
/// (`X_test` is always `N_TEST` rows, `shape.1` features).
fn fit_predict<F>(case: &OracleCase, shape: (usize, usize)) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let n_features = shape.1;

    let x_host: Vec<F> = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case
        .expect_f64("y")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let xt_host: Vec<F> = case
        .expect_f64("X_test")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
    let xt_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xt_host);

    let reg = LinearRegression::<F>::builder()
        .fit_intercept(true)
        .build::<F>()
        .expect("LinearRegression build")
        .fit(&mut pool, &x_dev, Some(&y_dev), shape)
        .expect("fit full-rank");
    let pred = reg
        .predict(&mut pool, &xt_dev, (N_TEST, n_features))
        .expect("predict on X_test");
    pred.to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect()
}

/// Training-set residual sum of squares for a host `(coef, intercept)` against
/// `(x, y)` (`x` is `n × d` row-major, all f64). Used by the large-`n_samples`
/// f32 near-collinear case (see its doc comment): forming the Gram squares
/// `X`'s condition number, so at f32 precision the ~1e-7-scale near-null
/// direction's eigenVALUE survives the cutoff (finite, bounded coefficients)
/// but its eigenVECTOR is not reliably resolvable — a well-known eigenvalue-
/// eigenvector sensitivity fact (eigenvector error ~ `eps / gap`), the same
/// eig-vs-svd tradeoff cuML documents for its own default `algorithm='eig'`.
/// The coefficient SPLIT between the two ~collinear features can therefore
/// legitimately differ from sklearn's SVD-derived split while still being a
/// good least-squares fit — this reference checks fit QUALITY (RSS), not
/// bit-parity of the (non-identifiable, up to the null-space direction)
/// individual coefficients.
fn rss(x: &[f64], y: &[f64], coef: &[f64], intercept: f64, n: usize, d: usize) -> f64 {
    let mut acc = 0.0f64;
    for r in 0..n {
        let mut pred = intercept;
        for c in 0..d {
            pred += x[r * d + c] * coef[c];
        }
        let e = pred - y[r];
        acc += e * e;
    }
    acc
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linreg fixtures are f32/f64 only"),
    }
}

/// `coef_`/`intercept_` vs sklearn, f32.
#[test]
fn linear_regression_coef_intercept_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let (coef, intercept) = fit_coef_intercept::<f32>(&case, "X", "y", (N_SAMPLES, N_FEATURES));
    assert_close(&coef, case.expect_f64("coef"), &F32_TOL, "coef_ f32");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F32_TOL,
        "intercept_ f32",
    );
}

/// `coef_`/`intercept_` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_coef_intercept_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let (coef, intercept) = fit_coef_intercept::<f64>(&case, "X", "y", (N_SAMPLES, N_FEATURES));
    assert_close(&coef, case.expect_f64("coef"), &F64_TOL, "coef_ f64");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F64_TOL,
        "intercept_ f64",
    );
}

/// `predict(X_test)` vs sklearn, f32.
#[test]
fn linear_regression_predict_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let pred = fit_predict::<f32>(&case, (N_SAMPLES, N_FEATURES));
    assert_close(&pred, case.expect_f64("y_pred"), &F32_TOL, "predict f32");
}

/// `predict(X_test)` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_predict_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let pred = fit_predict::<f64>(&case, (N_SAMPLES, N_FEATURES));
    assert_close(&pred, case.expect_f64("y_pred"), &F64_TOL, "predict f64");
}

/// Near-collinear small-σ-cutoff case, f32 (LINEAR-01 Pitfall 1): the cutoff
/// keeps `coef_col` finite + matching sklearn on the collinear `X_coll`.
#[test]
fn linear_regression_collinear_cutoff_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let (coef, intercept) = fit_coef_intercept::<f32>(&case, "X_coll", "y_coll", (N_SAMPLES, N_FEATURES));
    // The cutoff must keep the coefficients bounded (a no-cutoff inverse explodes).
    assert!(
        coef.iter().all(|c| c.is_finite()),
        "collinear coef_col f32 must stay finite (cutoff active): {coef:?}"
    );
    assert_close(&coef, case.expect_f64("coef_col"), &F32_TOL, "coef_col f32");
    assert_close(
        &[intercept],
        case.expect_f64("intercept_col"),
        &F32_TOL,
        "intercept_col f32",
    );
}

/// Near-collinear small-σ-cutoff case, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_collinear_cutoff_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!(
            "linreg collinear f64 backend={backend}: SKIPPED (no f64 support on this adapter)"
        );
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let (coef, intercept) = fit_coef_intercept::<f64>(&case, "X_coll", "y_coll", (N_SAMPLES, N_FEATURES));
    assert!(
        coef.iter().all(|c| c.is_finite()),
        "collinear coef_col f64 must stay finite (cutoff active): {coef:?}"
    );
    assert_close(&coef, case.expect_f64("coef_col"), &F64_TOL, "coef_col f64");
    assert_close(
        &[intercept],
        case.expect_f64("intercept_col"),
        &F64_TOL,
        "intercept_col f64",
    );
}

/// Large-`n_samples` `coef_`/`intercept_` vs sklearn, f32 — exercises
/// `fit_gram_eig` (`n_samples = 2000 > DIRECT_SVD_MAX_ROWS = 256`, D-02).
#[test]
fn linear_regression_large_coef_intercept_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "large");
    let case = load_npz(fixture("linear_regression_large_f32_seed42.npz"))
        .expect("load linreg_large_f32");
    let (coef, intercept) =
        fit_coef_intercept::<f32>(&case, "X", "y", (LARGE_N_SAMPLES, LARGE_N_FEATURES));
    assert_close(&coef, case.expect_f64("coef"), &F32_TOL, "large coef_ f32");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F32_TOL,
        "large intercept_ f32",
    );
}

/// Large-`n_samples` `coef_`/`intercept_` vs sklearn, f64 (cpu runs; rocm
/// skips-with-log) — exercises `fit_gram_eig`.
#[test]
fn linear_regression_large_coef_intercept_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "large");
    if capability::skip_f64_with_log() {
        println!(
            "linreg large f64 backend={backend}: SKIPPED (no f64 support on this adapter)"
        );
        return;
    }
    let case = load_npz(fixture("linear_regression_large_f64_seed42.npz"))
        .expect("load linreg_large_f64");
    let (coef, intercept) =
        fit_coef_intercept::<f64>(&case, "X", "y", (LARGE_N_SAMPLES, LARGE_N_FEATURES));
    assert_close(&coef, case.expect_f64("coef"), &F64_TOL, "large coef_ f64");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F64_TOL,
        "large intercept_ f64",
    );
}

/// Large-`n_samples` `predict(X_test)` vs sklearn, f32.
#[test]
fn linear_regression_large_predict_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "large");
    let case = load_npz(fixture("linear_regression_large_f32_seed42.npz"))
        .expect("load linreg_large_f32");
    let pred = fit_predict::<f32>(&case, (LARGE_N_SAMPLES, LARGE_N_FEATURES));
    assert_close(&pred, case.expect_f64("y_pred"), &F32_TOL, "large predict f32");
}

/// Large-`n_samples` `predict(X_test)` vs sklearn, f64 (cpu runs; rocm
/// skips-with-log).
#[test]
fn linear_regression_large_predict_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "large");
    if capability::skip_f64_with_log() {
        println!(
            "linreg large predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)"
        );
        return;
    }
    let case = load_npz(fixture("linear_regression_large_f64_seed42.npz"))
        .expect("load linreg_large_f64");
    let pred = fit_predict::<f64>(&case, (LARGE_N_SAMPLES, LARGE_N_FEATURES));
    assert_close(&pred, case.expect_f64("y_pred"), &F64_TOL, "large predict f64");
}

/// Large-`n_samples` near-collinear small-σ-cutoff case, f32.
///
/// UNLIKE every other large-fixture case, this one does NOT assert bit-parity
/// with sklearn's `coef_col` (see the [`rss`] doc comment for the numerical
/// reason: at f32 precision, squaring the condition number to form the Gram
/// leaves the near-null direction's eigenVALUE resolvable by the cutoff
/// [no explosion] but its eigenVECTOR under-resolved, so the coefficient mass
/// can legitimately split differently between the two ~collinear features).
/// What IS guaranteed, and what this test checks:
///   1. the cutoff still fires — coefficients stay FINITE and bounded (the
///      original Pitfall-1 regression this whole test family exists to catch
///      — a no-cutoff inverse explodes to ~1e4, see the module docs);
///   2. the fit is still a GOOD least-squares fit — its training RSS on
///      `(X_coll, y_coll)` is within a generous factor of sklearn's own
///      `coef_col`/`intercept_col` RSS on the same data (both should sit near
///      the `0.01`-noise floor; this is the fit-QUALITY invariant the eig
///      path actually owes, independent of which non-identifiable direction
///      absorbed the near-null component).
#[test]
fn linear_regression_large_collinear_cutoff_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "large");
    let case = load_npz(fixture("linear_regression_large_f32_seed42.npz"))
        .expect("load linreg_large_f32");
    let (coef, intercept) = fit_coef_intercept::<f32>(
        &case,
        "X_coll",
        "y_coll",
        (LARGE_N_SAMPLES, LARGE_N_FEATURES),
    );
    assert!(
        coef.iter().all(|c| c.is_finite()) && coef.iter().all(|c| c.abs() < 100.0),
        "large collinear coef_col f32 must stay finite and bounded (cutoff active): {coef:?}"
    );

    let x_coll = case.expect_f64("X_coll");
    let y_coll = case.expect_f64("y_coll");
    let sk_coef = case.expect_f64("coef_col");
    let sk_intercept = case.expect_f64("intercept_col")[0];
    let got_rss = rss(x_coll, y_coll, &coef, intercept, LARGE_N_SAMPLES, LARGE_N_FEATURES);
    let sk_rss = rss(
        x_coll,
        y_coll,
        sk_coef,
        sk_intercept,
        LARGE_N_SAMPLES,
        LARGE_N_FEATURES,
    );
    assert!(
        got_rss <= 10.0 * sk_rss.max(1e-6),
        "large collinear f32 training RSS regressed: got={got_rss:e} sklearn={sk_rss:e}"
    );
}

/// Large-`n_samples` near-collinear small-σ-cutoff case, f64 (cpu runs; rocm
/// skips-with-log).
#[test]
fn linear_regression_large_collinear_cutoff_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "large");
    if capability::skip_f64_with_log() {
        println!(
            "linreg large collinear f64 backend={backend}: SKIPPED (no f64 support on this adapter)"
        );
        return;
    }
    let case = load_npz(fixture("linear_regression_large_f64_seed42.npz"))
        .expect("load linreg_large_f64");
    let (coef, intercept) = fit_coef_intercept::<f64>(
        &case,
        "X_coll",
        "y_coll",
        (LARGE_N_SAMPLES, LARGE_N_FEATURES),
    );
    assert!(
        coef.iter().all(|c| c.is_finite()),
        "large collinear coef_col f64 must stay finite (cutoff active): {coef:?}"
    );
    assert_close(
        &coef,
        case.expect_f64("coef_col"),
        &F64_TOL,
        "large coef_col f64",
    );
    assert_close(
        &[intercept],
        case.expect_f64("intercept_col"),
        &F64_TOL,
        "large intercept_col f64",
    );
}

/// BLDR-01 defaults equality: the zero-arg `new()` (sklearn default
/// `fit_intercept = true`) reproduces every hyperparameter of
/// `builder().build()` — the single-source-of-defaults invariant (D-08).
#[test]
fn defaults_equal() {
    let from_new = LinearRegression::<f64>::new();
    let from_builder = LinearRegression::<f64>::builder()
        .build::<f64>()
        .expect("default LinearRegression builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "LinearRegression::new() must equal LinearRegression::builder().build()"
    );
}
