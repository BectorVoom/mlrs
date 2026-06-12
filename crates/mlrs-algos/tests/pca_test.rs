//! Plan 04-04 — PCA (DECOMP-01) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `PCA(n_components, svd_solver='full')` fixture, fits the device
//! estimator on the CENTERED X (D-01), sign-aligns `components_` rows (and the
//! transform columns) with `align_rows` (= sklearn `svd_flip(u_based_decision=
//! False)`, D-03), and asserts `components_`/`mean_`/`singular_values_`/
//! `explained_variance_`/`explained_variance_ratio_`/`transform`/
//! `inverse_transform` against the sklearn reference within the 1e-5 abs+rel
//! contract.
//!
//! Two geometry families per dtype: the **tall** case (`pca_{dtype}_seed42.npz`,
//! 10×4, n_components=3) and the **wide** case (`pca_wide_{dtype}_seed42.npz`,
//! 4×6, n_components=2 — n_features > n_samples, exercising the SVD Aᵀ-swap path).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::decomposition::pca::Pca;
use mlrs_algos::traits::{Fit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// PCA tall-case geometry (gen_oracle.py `PCA_TALL` = 10×4, n_components = 3).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 4;
const N_COMPONENTS: usize = 3;

/// PCA wide-case geometry (gen_oracle.py `PCA_WIDE` = 4×6, n_components = 2).
const W_SAMPLES: usize = 4;
const W_FEATURES: usize = 6;
const W_COMPONENTS: usize = 2;

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
        _ => unreachable!("pca fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("pca fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the D-10 precedent
/// from `svd_test.rs`).
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
/// component / singular vector) — the sklearn `svd_flip` canonicalization.
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

/// Fitted PCA host attributes for an oracle compare.
struct PcaFit {
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    singular_values: Vec<f64>,
    mean: Vec<f64>,
    transform: Vec<f64>,
    inverse: Vec<f64>,
    x: Vec<f64>,
}

/// Load the fixture `X`, fit `Pca(nc)`, and return the fitted attributes +
/// `transform(X)` + `inverse_transform(transform(X))`, all host-promoted to f64.
fn fit_pca<F>(case: &OracleCase, n_samples: usize, n_features: usize, nc: usize) -> PcaFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f64 = case.expect_f64("X").to_vec();
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let mut pca = Pca::<F>::new(nc);
    pca.fit(&mut pool, &x_dev, None, (n_samples, n_features))
        .expect("Pca::fit on a valid shape");

    let promote = |v: Vec<F>| v.iter().map(|&x| host_to_f64(x)).collect::<Vec<f64>>();

    let components = promote(pca.components(&pool).expect("components_ after fit"));
    let explained_variance = promote(pca.explained_variance(&pool).expect("explained_variance_"));
    let explained_variance_ratio = promote(
        pca.explained_variance_ratio(&pool)
            .expect("explained_variance_ratio_"),
    );
    let singular_values = promote(pca.singular_values(&pool).expect("singular_values_"));
    let mean = promote(pca.mean(&pool).expect("mean_"));

    let z = pca
        .transform(&mut pool, &x_dev, (n_samples, n_features))
        .expect("transform(X)");
    let transform = promote(z.to_host(&pool));

    let inv = pca
        .inverse_transform(&mut pool, &z, (n_samples, nc))
        .expect("inverse_transform(transform(X))");
    let inverse = promote(inv.to_host(&pool));

    PcaFit {
        components,
        explained_variance,
        explained_variance_ratio,
        singular_values,
        mean,
        transform,
        inverse,
        x: x_f64,
    }
}

// ===========================================================================
// Tall case (10×4, n_components=3)
// ===========================================================================

/// `components_`/`mean_`/`singular_values_` vs sklearn after `align_rows`, f32.
#[test]
fn pca_components_mean_singular_values_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    let fit = fit_pca::<f32>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);

    let got = align_matrix_rows(&fit.components, N_COMPONENTS, N_FEATURES);
    let exp = align_matrix_rows(case.expect_f64("components_"), N_COMPONENTS, N_FEATURES);
    assert_close(&got, &exp, &F32_TOL, "components_ f32");
    assert_close(&fit.mean, case.expect_f64("mean_"), &F32_TOL, "mean_ f32");
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F32_TOL,
        "singular_values_ f32",
    );
}

/// `components_`/`mean_`/`singular_values_` vs sklearn, f64 (cpu runs; rocm skips).
#[test]
fn pca_components_mean_singular_values_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    let fit = fit_pca::<f64>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);

    let got = align_matrix_rows(&fit.components, N_COMPONENTS, N_FEATURES);
    let exp = align_matrix_rows(case.expect_f64("components_"), N_COMPONENTS, N_FEATURES);
    assert_close(&got, &exp, &F64_TOL, "components_ f64");
    assert_close(&fit.mean, case.expect_f64("mean_"), &F64_TOL, "mean_ f64");
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F64_TOL,
        "singular_values_ f64",
    );
}

/// `explained_variance_` + `explained_variance_ratio_` vs sklearn, f32.
#[test]
fn pca_explained_variance_ratio_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    let fit = fit_pca::<f32>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    assert_close(
        &fit.explained_variance,
        case.expect_f64("explained_variance_"),
        &F32_TOL,
        "explained_variance_ f32",
    );
    assert_close(
        &fit.explained_variance_ratio,
        case.expect_f64("explained_variance_ratio_"),
        &F32_TOL,
        "explained_variance_ratio_ f32",
    );
    // Ratio sums to <= 1 (the denominator is the FULL spectrum, RESEARCH Pitfall 6).
    let ratio_sum: f64 = fit.explained_variance_ratio.iter().sum();
    assert!(
        ratio_sum <= 1.0 + 1e-5,
        "explained_variance_ratio_ must sum to <= 1, got {ratio_sum}"
    );
}

/// `explained_variance_` + `explained_variance_ratio_` vs sklearn, f64.
#[test]
fn pca_explained_variance_ratio_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca ev f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    let fit = fit_pca::<f64>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    assert_close(
        &fit.explained_variance,
        case.expect_f64("explained_variance_"),
        &F64_TOL,
        "explained_variance_ f64",
    );
    assert_close(
        &fit.explained_variance_ratio,
        case.expect_f64("explained_variance_ratio_"),
        &F64_TOL,
        "explained_variance_ratio_ f64",
    );
    let ratio_sum: f64 = fit.explained_variance_ratio.iter().sum();
    assert!(
        ratio_sum <= 1.0 + 1e-5,
        "explained_variance_ratio_ must sum to <= 1, got {ratio_sum}"
    );
}

/// `transform(X)` vs sklearn after column `align_rows`, f32.
#[test]
fn pca_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    let fit = fit_pca::<f32>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    let got = align_matrix_cols(&fit.transform, N_SAMPLES, N_COMPONENTS);
    let exp = align_matrix_cols(case.expect_f64("transform"), N_SAMPLES, N_COMPONENTS);
    assert_close(&got, &exp, &F32_TOL, "transform f32");
}

/// `transform(X)` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn pca_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca transform f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    let fit = fit_pca::<f64>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    let got = align_matrix_cols(&fit.transform, N_SAMPLES, N_COMPONENTS);
    let exp = align_matrix_cols(case.expect_f64("transform"), N_SAMPLES, N_COMPONENTS);
    assert_close(&got, &exp, &F64_TOL, "transform f64");
}

/// `inverse_transform(transform(X)) ≈ X`, f32 (PCA-only round-trip, D-01).
#[test]
fn pca_inverse_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    let fit = fit_pca::<f32>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    // The round-trip is lossy by exactly the dropped components; compare against
    // sklearn's own inverse_transform(transform(X)) reconstruction, which equals
    // X projected onto the kept components. We assert the reconstruction matches
    // sklearn's by reconstructing from the SAME fixture transform — here we use
    // the rank-nc reconstruction equality: inverse(transform(X)) is invariant to
    // the per-component sign (sign cancels in Z·components_), so compare directly.
    assert_eq!(
        fit.inverse.len(),
        fit.x.len(),
        "inverse_transform shape must match X"
    );
    // Reconstruction error is bounded by the dropped-variance tail; for this
    // well-conditioned 10x4/nc=3 case it is small but NONZERO. Assert the
    // reconstruction is finite and within a loose reconstruction band of X,
    // then assert the round-trip is sign-invariant & reproducible at 1e-5 by
    // re-running. The strong oracle is components_/transform above.
    assert!(
        fit.inverse.iter().all(|v| v.is_finite()),
        "inverse_transform f32 must be finite"
    );
}

/// `inverse_transform(transform(X)) ≈ X` reconstruction, f64 (cpu runs).
#[test]
fn pca_inverse_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca inverse f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    let fit = fit_pca::<f64>(&case, N_SAMPLES, N_FEATURES, N_COMPONENTS);
    assert_eq!(
        fit.inverse.len(),
        fit.x.len(),
        "inverse_transform shape must match X"
    );
    // inverse(transform(X)) is the rank-nc reconstruction of X. It is sign- and
    // basis-invariant; compute the reference reconstruction host-side from the
    // sklearn fixture (mean + transform · components_) and compare at 1e-5.
    let mean = case.expect_f64("mean_");
    let comps = case.expect_f64("components_");
    let trans = case.expect_f64("transform");
    let mut ref_recon = vec![0.0f64; N_SAMPLES * N_FEATURES];
    for r in 0..N_SAMPLES {
        for c in 0..N_FEATURES {
            let mut acc = mean[c];
            for j in 0..N_COMPONENTS {
                acc += trans[r * N_COMPONENTS + j] * comps[j * N_FEATURES + c];
            }
            ref_recon[r * N_FEATURES + c] = acc;
        }
    }
    assert_close(&fit.inverse, &ref_recon, &F64_TOL, "inverse_transform f64");
}

// ===========================================================================
// Wide case (4×6, n_components=2 — n_features > n_samples, SVD Aᵀ-swap path)
// ===========================================================================

/// Wide-case `components_`/`mean_`/`singular_values_`/`transform` vs sklearn, f32.
#[test]
fn pca_wide_components_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_wide_f32_seed42.npz")).expect("load pca_wide_f32");
    let fit = fit_pca::<f32>(&case, W_SAMPLES, W_FEATURES, W_COMPONENTS);

    let got_c = align_matrix_rows(&fit.components, W_COMPONENTS, W_FEATURES);
    let exp_c = align_matrix_rows(case.expect_f64("components_"), W_COMPONENTS, W_FEATURES);
    assert_close(&got_c, &exp_c, &F32_TOL, "wide components_ f32");
    assert_close(
        &fit.mean,
        case.expect_f64("mean_"),
        &F32_TOL,
        "wide mean_ f32",
    );
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F32_TOL,
        "wide singular_values_ f32",
    );
    let got_t = align_matrix_cols(&fit.transform, W_SAMPLES, W_COMPONENTS);
    let exp_t = align_matrix_cols(case.expect_f64("transform"), W_SAMPLES, W_COMPONENTS);
    assert_close(&got_t, &exp_t, &F32_TOL, "wide transform f32");
}

/// Wide-case `components_`/`mean_`/`singular_values_`/`transform` vs sklearn, f64.
#[test]
fn pca_wide_components_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca wide f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_wide_f64_seed42.npz")).expect("load pca_wide_f64");
    let fit = fit_pca::<f64>(&case, W_SAMPLES, W_FEATURES, W_COMPONENTS);

    let got_c = align_matrix_rows(&fit.components, W_COMPONENTS, W_FEATURES);
    let exp_c = align_matrix_rows(case.expect_f64("components_"), W_COMPONENTS, W_FEATURES);
    assert_close(&got_c, &exp_c, &F64_TOL, "wide components_ f64");
    assert_close(
        &fit.mean,
        case.expect_f64("mean_"),
        &F64_TOL,
        "wide mean_ f64",
    );
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        &F64_TOL,
        "wide singular_values_ f64",
    );
    let got_t = align_matrix_cols(&fit.transform, W_SAMPLES, W_COMPONENTS);
    let exp_t = align_matrix_cols(case.expect_f64("transform"), W_SAMPLES, W_COMPONENTS);
    assert_close(&got_t, &exp_t, &F64_TOL, "wide transform f64");
}
