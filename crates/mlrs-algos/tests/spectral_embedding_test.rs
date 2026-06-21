//! Plan 09-03 — SpectralEmbedding (SPECTRAL-01) sklearn oracle tests.
//!
//! Activated from the 09-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `SpectralEmbedding` fixture, fits the device estimator, and
//! value-matches (after sign alignment) or subspace-matches sklearn's
//! `embedding_`. The pipeline is affinity → normalized Laplacian → v1 `eig`
//! (DESCENDING, reversed to ascending) → `/dd` recovery (D-07) →
//! `_deterministic_vector_sign_flip` → drop the trivial row 0 (drop_first, D-08).
//!
//! Case map (9-SE-01..04):
//!   - `spectral_embedding` — rbf affinity (gamma=None→1/n_features, D-02/D-04)
//!     value-match after sign alignment. The RESEARCH-validated dense
//!     full-spectrum path (reproduces sklearn ARPACK to ~1e-15 here); f64 strict.
//!   - `knn_affinity` — `nearest_neighbors` connectivity affinity (D-03) with the
//!     fixture's explicit connected `n_neighbors`, value-match after sign align.
//!   - `subspace` — degenerate-spectrum subspace test (principal angles, D-09):
//!     the cycle-graph fixture has a degenerate Fiedler pair, so the kept
//!     eigenspace matches sklearn as a COLUMN SPACE (not per element).
//!   - `reject_oversize` — `n_samples > 64` → `AlgoError::NSamplesExceedsMaxDim`
//!     BEFORE any device work (D-06): a live `fit(n=65)` rejection.
//!
//! f64 carries the `skip_f64_with_log` gate verbatim; f32 runs at the documented
//! `SE_F32_BAND` (~1e-4, Pitfall 7). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::SpectralEmbedding;
use mlrs_algos::error::AlgoError;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F64_TOL};

/// SpectralEmbedding fixture geometry (gen_oracle.py `SE_N_SAMPLES` ×
/// `SE_N_FEATURES`, `SE_N_COMPONENTS` embedding columns).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 5;
const N_COMPONENTS: usize = 2;

/// Documented f32 band for the SPECTRAL-01 embedding (the v1 per-family
/// documented-band precedent; the strict 1e-5 absolute arm is never loosened).
/// f64 stays strict `F64_TOL` (1e-5). The observed max f32 error is recorded in
/// the SUMMARY.
const SE_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

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
        _ => unreachable!("spectral_embedding fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("spectral_embedding fixtures are f32/f64 only"),
    }
}

/// Fit a `SpectralEmbedding` of the requested affinity on the fixture's `X` and
/// return the host `embedding_` (row-major `n × n_components`).
fn fit_embedding<F>(case: &OracleCase, affinity: &str, n_neighbors: usize) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    // gamma=None → 1/n_features resolved at fit (D-04). The rbf path uses it; the
    // nearest_neighbors path ignores it.
    let mut se = SpectralEmbedding::<F>::new(
        N_COMPONENTS,
        affinity.to_string(),
        None,
        n_neighbors,
    );
    se.fit(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("SpectralEmbedding::fit on a valid shape");

    se.embedding(&pool)
        .expect("embedding_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect()
}

/// Column-wise sign-aligned `allclose`: each embedding column is defined only up
/// to a global sign, so align the sign of `got[:,c]` to `expected[:,c]` (by the
/// sign of their dot product) before the strict abs-OR-rel compare. Returns the
/// observed max abs error for SUMMARY-band documentation.
fn assert_close_sign_aligned(
    got: &[f64],
    expected: &[f64],
    n: usize,
    k: usize,
    tol: &Tolerance,
    what: &str,
) -> f64 {
    assert_eq!(got.len(), n * k, "{what}: got length mismatch");
    assert_eq!(expected.len(), n * k, "{what}: expected length mismatch");

    let mut max_abs = 0.0f64;
    for c in 0..k {
        // Sign-align column c.
        let mut dot = 0.0f64;
        for i in 0..n {
            dot += got[i * k + c] * expected[i * k + c];
        }
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            let g = sign * got[i * k + c];
            let e = expected[i * k + c];
            assert!(g.is_finite(), "{what}: non-finite got at ({i},{c}): {g:e}");
            let abs_err = (g - e).abs();
            max_abs = max_abs.max(abs_err);
            let allclose = abs_err <= tol.abs + tol.rel * e.abs();
            assert!(
                allclose,
                "{what}: allclose failed at ({i},{c}): got={g:e} expected={e:e} \
                 abs_err={abs_err:e} (atol={:e}, rtol={:e})",
                tol.abs, tol.rel
            );
        }
    }
    max_abs
}

/// Orthonormalize the `k` columns of a row-major `n × k` matrix via classical
/// Gram–Schmidt, returning the row-major `n × k` orthonormal basis `Q`.
fn orthonormalize(m: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut q = vec![0.0f64; n * k];
    for c in 0..k {
        // Start from column c.
        let mut v: Vec<f64> = (0..n).map(|i| m[i * k + c]).collect();
        // Subtract projections onto the earlier orthonormal columns.
        for prev in 0..c {
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += v[i] * q[i * k + prev];
            }
            for i in 0..n {
                v[i] -= dot * q[i * k + prev];
            }
        }
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-12, "orthonormalize: degenerate column {c}");
        for i in 0..n {
            q[i * k + c] = v[i] / norm;
        }
    }
    q
}

/// Subspace-distance test via principal angles (D-09). For two `n × k`
/// embeddings, orthonormalize each column space (`Q1`, `Q2`), form `M = Q1ᵀ Q2`
/// (`k × k`), and the cosines of the principal angles are the singular values of
/// `M`. Identical column spaces ⇒ all singular values ≈ 1. We assert the SMALLEST
/// singular value of `M` is ≥ `1 - tol` (the largest principal angle ≈ 0).
/// Returns `1 - σ_min` (the subspace mismatch) for SUMMARY documentation.
fn subspace_mismatch(got: &[f64], expected: &[f64], n: usize, k: usize) -> f64 {
    assert_eq!(k, 2, "subspace_mismatch is specialized to k=2 (SE n_components)");
    let q1 = orthonormalize(got, n, k);
    let q2 = orthonormalize(expected, n, k);

    // M = Q1ᵀ Q2 (k × k = 2 × 2).
    let mut mm = [[0.0f64; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            let mut s = 0.0f64;
            for i in 0..n {
                s += q1[i * k + a] * q2[i * k + b];
            }
            mm[a][b] = s;
        }
    }
    // Singular values of the 2×2 M: σ² are the eigenvalues of MᵀM.
    let m00 = mm[0][0];
    let m01 = mm[0][1];
    let m10 = mm[1][0];
    let m11 = mm[1][1];
    let a = m00 * m00 + m10 * m10; // (MᵀM)[0,0]
    let b = m00 * m01 + m10 * m11; // (MᵀM)[0,1] = [1,0]
    let d = m01 * m01 + m11 * m11; // (MᵀM)[1,1]
    let trace = a + d;
    let det = a * d - b * b;
    let disc = (trace * trace / 4.0 - det).max(0.0).sqrt();
    let lambda_min = (trace / 2.0 - disc).max(0.0);
    let sigma_min = lambda_min.sqrt();
    1.0 - sigma_min
}

/// 9-SE-01: rbf-affinity embedding value-match after sign alignment, f64 strict.
/// Gated by `skip_f64_with_log`.
#[test]
fn spectral_embedding() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_f64_seed42.npz"))
        .expect("load spectral_embedding_f64");
    let got = fit_embedding::<f64>(&case, "rbf", 0);
    let max_abs = assert_close_sign_aligned(
        &got,
        case.expect_f64("embedding"),
        N_SAMPLES,
        N_COMPONENTS,
        &F64_TOL,
        "spectral_embedding rbf f64",
    );
    println!("spectral_embedding rbf f64 max_abs_err = {max_abs:e}");
    let _ = &SE_F32_BAND; // band kept load-bearing for the f32 path below.
}

/// 9-SE-01 (f32): rbf-affinity embedding at the documented `SE_F32_BAND`.
#[test]
fn spectral_embedding_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("spectral_embedding_f32_seed42.npz"))
        .expect("load spectral_embedding_f32");
    let got = fit_embedding::<f32>(&case, "rbf", 0);
    let max_abs = assert_close_sign_aligned(
        &got,
        case.expect_f64("embedding"),
        N_SAMPLES,
        N_COMPONENTS,
        &SE_F32_BAND,
        "spectral_embedding rbf f32",
    );
    println!(
        "spectral_embedding rbf f32 max_abs_err = {max_abs:e} (band atol={:e})",
        SE_F32_BAND.abs
    );
    assert!(
        max_abs <= SE_F32_BAND.abs,
        "f32 max_abs_err {max_abs:e} exceeds documented band {:e}",
        SE_F32_BAND.abs
    );
}

/// 9-SE-02: `nearest_neighbors` connectivity-affinity embedding (D-01/D-03),
/// f64 strict. The fixture pins an explicit connected `n_neighbors`.
#[test]
fn knn_affinity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding knn f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_f64_seed42.npz"))
        .expect("load spectral_embedding_f64");
    let n_neighbors = case.expect_f64("n_neighbors")[0] as usize;
    let got = fit_embedding::<f64>(&case, "nearest_neighbors", n_neighbors);
    let max_abs = assert_close_sign_aligned(
        &got,
        case.expect_f64("embedding_knn"),
        N_SAMPLES,
        N_COMPONENTS,
        &F64_TOL,
        "spectral_embedding knn f64",
    );
    println!("spectral_embedding knn f64 max_abs_err = {max_abs:e}");
}

/// 9-SE-03: degenerate-spectrum subspace test (principal angles, D-09). The
/// cycle-graph fixture has a degenerate Fiedler pair, so the kept eigenspace is
/// defined only up to rotation: a per-vector value match would false-fail, but
/// the COLUMN SPACE matches sklearn. f64 strict on the subspace mismatch.
#[test]
fn subspace() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding subspace f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_degenerate_f64_seed42.npz"))
        .expect("load spectral_embedding_degenerate_f64");
    let got = fit_embedding::<f64>(&case, "rbf", 0);
    let expected = case.expect_f64("embedding");
    let mismatch = subspace_mismatch(&got, expected, N_SAMPLES, N_COMPONENTS);
    println!("spectral_embedding subspace f64 mismatch (1 - σ_min) = {mismatch:e}");
    assert!(
        mismatch <= 1e-5,
        "degenerate column space mismatch {mismatch:e} exceeds 1e-5 (principal \
         angle too large — the kept eigenspace does not match sklearn)"
    );
}

/// 9-SE-04: `n_samples > 64` is rejected with `AlgoError::NSamplesExceedsMaxDim`
/// BEFORE any device work (D-06). A live `fit(n=65)` must return the typed
/// spectral-cap error without any affinity / Laplacian / eig launch.
#[test]
fn reject_oversize() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // 65 samples > MAX_DIM (64). The buffer is the minimal n×d the geometry guard
    // accepts; the cap guard fires FIRST so no device kernel runs.
    let n = 65usize;
    let d = 3usize;
    let x_host: Vec<f64> = vec![0.0; n * d];
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let mut se =
        SpectralEmbedding::<f64>::new(N_COMPONENTS, "rbf".to_string(), None, 0);
    let err = match se.fit(&mut pool, &x_dev, (n, d)) {
        Ok(_) => panic!("fit(n=65) must reject before any device work"),
        Err(e) => e,
    };

    let msg = err.to_string();
    match err {
        AlgoError::NSamplesExceedsMaxDim {
            estimator,
            n_samples,
            max,
        } => {
            assert_eq!(estimator, "spectral_embedding");
            assert_eq!(n_samples, 65);
            assert_eq!(max, 64);
            assert!(
                msg.contains("65") && msg.contains("64") && msg.contains("MAX_DIM"),
                "NSamplesExceedsMaxDim message must name n_samples + the cap: {msg}"
            );
        }
        other => panic!("expected NSamplesExceedsMaxDim, got {other:?}"),
    }
}
