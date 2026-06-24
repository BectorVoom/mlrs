//! Plan 04-04 — TruncatedSVD (DECOMP-02) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `TruncatedSVD(n_components, algorithm='arpack')` fixture
//! (DETERMINISTIC, NOT randomized — D-07), fits the device estimator on the
//! UNCENTERED X (D-01), sign-aligns `components_` rows (and the transform columns)
//! with `align_rows` (= sklearn `svd_flip(u_based_decision=False)`, D-03), and
//! asserts `components_`/`singular_values_`/`explained_variance_`/`transform`
//! against the sklearn reference within the 1e-5 abs+rel contract.
//!
//! The `explained_variance_` here is the variance of the transformed columns
//! (population / ddof=0), NOT PCA's `S²/(n−1)` (RESEARCH Pitfall 2). The estimator
//! also does NOT center X.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::decomposition::truncated_svd::TruncatedSvd;
use mlrs_algos::typestate::{Fit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// TruncatedSVD geometry (gen_oracle.py `TSVD_SHAPE` = 10×5, n_components = 3).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 5;
const N_COMPONENTS: usize = 3;

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
        _ => unreachable!("tsvd fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("tsvd fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (D-10 precedent).
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

/// Sign-align a row-major `(rows × cols)` matrix by its ROWS (each row a
/// component) — the sklearn `svd_flip` canonicalization.
fn align_matrix_rows(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let row_vecs: Vec<Vec<f64>> = (0..rows)
        .map(|r| (0..cols).map(|c| mat[r * cols + c]).collect())
        .collect();
    let flipped = align_rows(&row_vecs);
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = flipped[r][c];
        }
    }
    out
}

/// Sign-align a row-major `(rows × cols)` matrix by its COLUMNS (each column the
/// projection onto one component) — the transform-side `svd_flip`.
fn align_matrix_cols(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let col_vecs: Vec<Vec<f64>> = (0..cols)
        .map(|c| (0..rows).map(|r| mat[r * cols + c]).collect())
        .collect();
    let flipped = align_rows(&col_vecs);
    let mut out = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            out[r * cols + c] = flipped[c][r];
        }
    }
    out
}

/// Fitted TruncatedSVD host attributes for an oracle compare.
struct TsvdFit {
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    singular_values: Vec<f64>,
    transform: Vec<f64>,
}

/// Load the fixture `X`, fit `TruncatedSvd(nc)`, and return the fitted attributes
/// + `transform(X)`, all host-promoted to f64.
fn fit_tsvd<F>(case: &OracleCase) -> TsvdFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let tsvd = TruncatedSvd::<F>::builder()
        .n_components(N_COMPONENTS)
        .build::<F>()
        .expect("TruncatedSvdBuilder::build is infallible")
        .fit(&mut pool, &x_dev, None, (N_SAMPLES, N_FEATURES))
        .expect("TruncatedSvd::fit on a valid shape");

    let promote = |v: Vec<F>| v.iter().map(|&x| host_to_f64(x)).collect::<Vec<f64>>();

    let components = promote(tsvd.components(&pool));
    let explained_variance = promote(tsvd.explained_variance(&pool));
    let singular_values = promote(tsvd.singular_values(&pool));

    let z = tsvd
        .transform(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("transform(X)");
    let transform = promote(z.to_host(&pool));

    TsvdFit {
        components,
        explained_variance,
        singular_values,
        transform,
    }
}

/// BLDR-01: the zero-arg `new()` defaults equal the builder's `build()` defaults
/// (`TruncatedSvd::new().hyperparams_eq(&TruncatedSvd::builder().build()?)`).
#[test]
fn truncated_svd_defaults_equal() {
    let from_new = TruncatedSvd::<f64>::new();
    let from_builder = TruncatedSvd::<f64>::builder()
        .build::<f64>()
        .expect("TruncatedSvdBuilder::build is infallible");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "TruncatedSvd::new() defaults must equal builder().build() defaults"
    );
}

/// `components_`/`singular_values_` vs sklearn arpack after `align_rows`, f32.
#[test]
fn truncated_svd_components_singular_values_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    let fit = fit_tsvd::<f32>(&case);
    let got = align_matrix_rows(&fit.components, N_COMPONENTS, N_FEATURES);
    let exp = align_matrix_rows(case.expect_f64("components_"), N_COMPONENTS, N_FEATURES);
    assert_close(&got, &exp, &F32_TOL, "components_ f32");
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F32_TOL,
        "singular_values_ f32",
    );
}

/// `components_`/`singular_values_` vs sklearn arpack, f64 (cpu runs; rocm skips).
#[test]
fn truncated_svd_components_singular_values_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    let fit = fit_tsvd::<f64>(&case);
    let got = align_matrix_rows(&fit.components, N_COMPONENTS, N_FEATURES);
    let exp = align_matrix_rows(case.expect_f64("components_"), N_COMPONENTS, N_FEATURES);
    assert_close(&got, &exp, &F64_TOL, "components_ f64");
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F64_TOL,
        "singular_values_ f64",
    );
}

/// `explained_variance_` (= var of transformed columns, Pitfall 2) vs sklearn, f32.
#[test]
fn truncated_svd_explained_variance_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    let fit = fit_tsvd::<f32>(&case);
    assert_close(
        &fit.explained_variance,
        case.expect_f64("explained_variance_"),
        &F32_TOL,
        "explained_variance_ f32",
    );
}

/// `explained_variance_` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn truncated_svd_explained_variance_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd ev f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    let fit = fit_tsvd::<f64>(&case);
    assert_close(
        &fit.explained_variance,
        case.expect_f64("explained_variance_"),
        &F64_TOL,
        "explained_variance_ f64",
    );
}

/// `transform(X)` vs sklearn arpack after column `align_rows`, f32.
#[test]
fn truncated_svd_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    let fit = fit_tsvd::<f32>(&case);
    let got = align_matrix_cols(&fit.transform, N_SAMPLES, N_COMPONENTS);
    let exp = align_matrix_cols(case.expect_f64("transform"), N_SAMPLES, N_COMPONENTS);
    assert_close(&got, &exp, &F32_TOL, "transform f32");
}

/// `transform(X)` vs sklearn arpack, f64 (cpu runs; rocm skips-with-log).
#[test]
fn truncated_svd_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd transform f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    let fit = fit_tsvd::<f64>(&case);
    let got = align_matrix_cols(&fit.transform, N_SAMPLES, N_COMPONENTS);
    let exp = align_matrix_cols(case.expect_f64("transform"), N_SAMPLES, N_COMPONENTS);
    assert_close(&got, &exp, &F64_TOL, "transform f64");
}
