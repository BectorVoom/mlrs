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
//! SPECTRAL-PERF-CPU adds the LANCZOS-arm cases. Everything above runs at n=12,
//! entirely on the dense `sym_eig` route, so the thick-restart Lanczos the host
//! pipeline uses above `spectral_host::DENSE_N` had no oracle at all:
//!   - `spectral_embedding_large_knn` — n=800, sparse kNN affinity, f64 strict.
//!   - `spectral_embedding_large_rbf` — n=700, dense rbf affinity, f64 strict.
//!   - `lanczos_matches_dense` — the two solvers on the SAME Laplacian at
//!     `n = DENSE_N + 8`, agreeing to ~1e-8 (solver isolation, no sklearn).
//!
//! f64 carries the `skip_f64_with_log` gate verbatim; f32 runs at the documented
//! `SE_F32_BAND` (~1e-4, Pitfall 7). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::SpectralEmbedding;
use mlrs_algos::typestate::Fit;
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
fn fit_embedding<F>(case: &OracleCase, affinity: &str, n_neighbors: Option<usize>) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    // gamma=None → 1/n_features resolved at fit (D-04). The rbf path uses it; the
    // nearest_neighbors path ignores it.
    let se = SpectralEmbedding::<F>::builder()
        .n_components(N_COMPONENTS)
        .affinity(affinity.to_string())
        .gamma(None)
        .n_neighbors(n_neighbors)
        .build::<F>()
        .expect("SpectralEmbedding build with valid hyperparameters");
    let se = se
        .fit(&mut pool, &x_dev, None, (N_SAMPLES, N_FEATURES))
        .expect("SpectralEmbedding::fit on a valid shape");

    se.embedding(&pool).iter().map(|&v| host_to_f64(v)).collect()
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
    let got = fit_embedding::<f64>(&case, "rbf", None);
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
    let got = fit_embedding::<f32>(&case, "rbf", None);
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
    let got = fit_embedding::<f64>(&case, "nearest_neighbors", Some(n_neighbors));
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
    let got = fit_embedding::<f64>(&case, "rbf", None);
    let expected = case.expect_f64("embedding");
    let mismatch = subspace_mismatch(&got, expected, N_SAMPLES, N_COMPONENTS);
    println!("spectral_embedding subspace f64 mismatch (1 - σ_min) = {mismatch:e}");
    assert!(
        mismatch <= 1e-5,
        "degenerate column space mismatch {mismatch:e} exceeds 1e-5 (principal \
         angle too large — the kept eigenspace does not match sklearn)"
    );
}

/// 9-SE-04 (REPLACED by SPECTRAL-PERF-CPU): `n_samples = 65` used to be rejected
/// with `AlgoError::NSamplesExceedsMaxDim`, because the dense cyclic-Jacobi `eig`
/// kernel stages `MAX_DIM x MAX_DIM` shared memory and so caps `n <= 64`. The
/// host pipeline has no such cap, and this test is the live proof: the SAME
/// `fit(n = 65)` that the old assertion required to fail must now SUCCEED and
/// return a finite `n x n_components` embedding.
///
/// The geometry is a 1-D lattice with a distinct coordinate per sample, so the
/// kNN graph is a connected path and the spectrum is non-degenerate.
#[test]
fn large_n_is_no_longer_capped() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 65usize;
    let d = 1usize;
    let x_host: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let se = SpectralEmbedding::<f64>::builder()
        .n_components(N_COMPONENTS)
        .affinity("nearest_neighbors".to_string())
        .n_neighbors(Some(5))
        .build::<f64>()
        .expect("SpectralEmbedding build with valid hyperparameters");
    let fitted = se
        .fit(&mut pool, &x_dev, None, (n, d))
        .expect("fit(n = 65) must succeed now that the MAX_DIM cap is gone");

    let emb = fitted.embedding(&pool);
    assert_eq!(emb.len(), n * N_COMPONENTS, "embedding_ must be n x n_components");
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "embedding_ must be finite"
    );
    // A path graph is connected, so the Laplacian has exactly one zero
    // eigenvalue and the kept (non-trivial) columns cannot be constant.
    assert_eq!(fitted.n_graph_components(), 1, "the path graph is connected");
    for c in 0..N_COMPONENTS {
        let col: Vec<f64> = (0..n).map(|i| emb[i * N_COMPONENTS + c]).collect();
        let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            hi - lo > 1e-9,
            "embedding column {c} is constant — the solver returned a trivial vector"
        );
    }
}

/// SPECTRAL-PERF-CPU: sklearn resolves `n_neighbors=None` to
/// `max(int(n_samples / 10), 1)` — TRUNCATING division, floored at 1. The
/// pre-rewrite implementation hard-coded `10`, so it built a DIFFERENT graph from
/// sklearn's on every input with `n_samples != 100`. Pin the resolution rule.
#[test]
fn n_neighbors_none_resolves_like_sklearn() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for (n, expect) in [(6usize, 1usize), (40, 4), (100, 10), (255, 25)] {
        let x_host: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
        let se = SpectralEmbedding::<f64>::builder()
            .n_components(1)
            .build::<f64>()
            .expect("default SpectralEmbedding build");
        let fitted = se
            .fit(&mut pool, &x_dev, None, (n, 1))
            .expect("fit with the default n_neighbors");
        assert_eq!(
            fitted.n_neighbors_(),
            Some(expect),
            "n_neighbors_ for n_samples = {n} must be max(n/10, 1) = {expect}"
        );
    }
}

// ---------------------------------------------------------------------------
// SPECTRAL-PERF-CPU — the LANCZOS arm (n_samples > spectral_host::DENSE_N)
// ---------------------------------------------------------------------------
//
// Every fixture above is n=12, i.e. entirely on the dense `sym_eig` route. The
// three tests below are the first coverage of the thick-restart Lanczos arm:
// two sklearn value-matches (one sparse kNN affinity, one dense rbf one) and a
// direct dense-vs-Lanczos equivalence check on a SINGLE shared operator.
//
// Both fixtures were generated only after `scripts/gen_oracle.py` verified, and
// asserted, that (a) the affinity graph is CONNECTED and (b) every consecutive
// gap over the kept part of the Laplacian spectrum exceeds 1e-3 — without both,
// the retained eigenspace is defined only up to a rotation and a per-element
// comparison against sklearn is meaningless. The verified spectra are committed
// in each fixture's `eigs` array and re-checked here.

/// `spectral_embedding_large_f64.npz` geometry (gen_oracle.py `SE_LARGE_*`).
const LARGE_N: usize = 800;
const LARGE_D: usize = 8;
const LARGE_COMPONENTS: usize = 3;

/// `spectral_embedding_large_rbf_f64.npz` geometry (gen_oracle.py
/// `SE_LARGE_RBF_*`).
const LARGE_RBF_N: usize = 700;
const LARGE_RBF_D: usize = 6;
const LARGE_RBF_COMPONENTS: usize = 2;

/// Smallest consecutive eigenvalue gap `gen_oracle.py` accepted over the kept
/// spectrum (`SE_LARGE_MIN_GAP`). Re-asserted from the committed `eigs` so a
/// regenerated fixture that quietly went degenerate fails HERE, loudly, instead
/// of producing a mysterious value mismatch.
const LARGE_MIN_GAP: f64 = 1e-3;

/// Fit a `SpectralEmbedding` on a fixture of arbitrary geometry and return the
/// host `embedding_` (row-major `n × n_components`).
///
/// Uses [`SpectralEmbedding::fit_from_host_slice`] — the no-upload arm the
/// estimator's own `host_fit_applicable` always selects — rather than the
/// `DeviceArray` `Fit::fit` the small fixtures above use; both funnel into the
/// same `fit_host_core`, and at n=800 there is no reason to pay the round trip.
fn fit_embedding_shaped(
    case: &OracleCase,
    affinity: &str,
    n_neighbors: Option<usize>,
    shape: (usize, usize),
    n_components: usize,
) -> Vec<f64> {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = case.expect_f64("X").to_vec();
    assert_eq!(
        x_host.len(),
        shape.0 * shape.1,
        "fixture X must be {} x {}",
        shape.0,
        shape.1
    );

    let se = SpectralEmbedding::<f64>::builder()
        .n_components(n_components)
        .affinity(affinity.to_string())
        // gamma=None → 1/n_features at fit (D-04); the kNN path ignores it.
        .gamma(None)
        .n_neighbors(n_neighbors)
        .build::<f64>()
        .expect("SpectralEmbedding build with valid hyperparameters");
    let se = se
        .fit_from_host_slice(&mut pool, &x_host, shape)
        .expect("SpectralEmbedding::fit_from_host_slice on a valid shape");

    assert_eq!(
        se.n_graph_components(),
        1,
        "the fixture's affinity graph must be connected (gen_oracle.py asserts \
         it too) — a disconnected graph makes the kept eigenspace ambiguous"
    );
    se.embedding(&pool)
}

/// Assert the committed Laplacian spectrum is non-degenerate over the kept
/// range, i.e. that the fixture is still a valid per-element oracle. Returns the
/// smallest observed gap for the printed record.
fn assert_spectrum_separated(eigs: &[f64], nev: usize, what: &str) -> f64 {
    assert!(
        eigs.len() >= nev + 1,
        "{what}: fixture must commit at least nev+1 = {} eigenvalues, got {}",
        nev + 1,
        eigs.len()
    );
    let mut min_gap = f64::INFINITY;
    for r in 0..nev {
        let gap = eigs[r + 1] - eigs[r];
        min_gap = min_gap.min(gap);
    }
    assert!(
        min_gap > LARGE_MIN_GAP,
        "{what}: smallest kept eigenvalue gap {min_gap:e} <= {LARGE_MIN_GAP:e} — \
         the retained eigenspace is (near-)degenerate and this fixture cannot be \
         value-matched element by element"
    );
    min_gap
}

/// SPECTRAL-PERF-CPU: the LANCZOS arm reproduces sklearn on a SPARSE kNN
/// affinity. `n_samples = 800 > DENSE_N = 512`, so `smallest_laplacian_vectors`
/// routes to `lanczos_largest`, not to the dense `sym_eig` every other spectral
/// fixture exercises. f64 strict `F64_TOL`.
#[test]
fn spectral_embedding_large_knn() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding_large knn f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    assert!(
        LARGE_N > mlrs_algos::cluster::spectral_host::DENSE_N,
        "the large kNN fixture must sit ABOVE the dense/Lanczos threshold"
    );
    let case = load_npz(fixture("spectral_embedding_large_f64.npz"))
        .expect("load spectral_embedding_large_f64");

    // drop_first=True → the solver is asked for n_components + 1 eigenvectors.
    let nev = LARGE_COMPONENTS + 1;
    let min_gap = assert_spectrum_separated(
        case.expect_f64("eigs"),
        nev,
        "spectral_embedding_large knn",
    );

    let n_neighbors = case.expect_f64("n_neighbors")[0] as usize;
    let got = fit_embedding_shaped(
        &case,
        "nearest_neighbors",
        Some(n_neighbors),
        (LARGE_N, LARGE_D),
        LARGE_COMPONENTS,
    );
    let max_abs = assert_close_sign_aligned(
        &got,
        case.expect_f64("embedding"),
        LARGE_N,
        LARGE_COMPONENTS,
        &F64_TOL,
        "spectral_embedding_large knn f64",
    );
    println!(
        "spectral_embedding_large knn f64 (n={LARGE_N}, k={n_neighbors}, Lanczos) \
         max_abs_err = {max_abs:e}, min eigenvalue gap = {min_gap:e}"
    );
}

/// SPECTRAL-PERF-CPU: the LANCZOS arm reproduces sklearn on a DENSE rbf
/// affinity — the same solver, but driving the dense matvec rather than the CSR
/// one. `n_samples = 700 > DENSE_N = 512`; `gamma=None → 1/n_features` (D-04).
#[test]
fn spectral_embedding_large_rbf() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding_large rbf f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    assert!(
        LARGE_RBF_N > mlrs_algos::cluster::spectral_host::DENSE_N,
        "the large rbf fixture must sit ABOVE the dense/Lanczos threshold"
    );
    let case = load_npz(fixture("spectral_embedding_large_rbf_f64.npz"))
        .expect("load spectral_embedding_large_rbf_f64");

    let nev = LARGE_RBF_COMPONENTS + 1;
    let min_gap = assert_spectrum_separated(
        case.expect_f64("eigs"),
        nev,
        "spectral_embedding_large rbf",
    );

    let got = fit_embedding_shaped(
        &case,
        "rbf",
        None,
        (LARGE_RBF_N, LARGE_RBF_D),
        LARGE_RBF_COMPONENTS,
    );
    let max_abs = assert_close_sign_aligned(
        &got,
        case.expect_f64("embedding"),
        LARGE_RBF_N,
        LARGE_RBF_COMPONENTS,
        &F64_TOL,
        "spectral_embedding_large rbf f64",
    );
    println!(
        "spectral_embedding_large rbf f64 (n={LARGE_RBF_N}, gamma=1/{LARGE_RBF_D}, \
         Lanczos) max_abs_err = {max_abs:e}, min eigenvalue gap = {min_gap:e}"
    );
}

/// SPECTRAL-PERF-CPU: the two solvers agree on the SAME operator.
///
/// The sklearn value tests above check the whole pipeline end to end; this one
/// isolates the solver. It builds ONE `NormAdj` at `n = DENSE_N + 8` — just past
/// the routing threshold, so it is exactly the regime where the choice flips —
/// and hands its Laplacian to BOTH arms:
///
/// - the dense route: `sym_eig(dense_laplacian)`, taking the `nev` SMALLEST
///   eigenpairs (columns `n-1-r`, since `sym_eig` is descending);
/// - the iterative route: `lanczos_largest`, whose `nev` largest eigenvectors of
///   `S = 2I − L` are those same eigenvectors.
///
/// Agreement to ~1e-8 is far tighter than the 1e-5 oracle band, and any real
/// defect in the restart, the arrow coupling, or the reorthogonalization shows
/// up here as a per-vector disagreement rather than as a diffuse end-to-end
/// error. The dense eigenvalues also gate the comparison: a degenerate pair
/// would make it vacuous, so the gaps are asserted first.
#[test]
fn lanczos_matches_dense() {
    use mlrs_algos::cluster::spectral_affinity::{build_affinity, AffinityKind};
    use mlrs_algos::cluster::spectral_host::{lanczos_largest, NormAdj, DENSE_N};
    use mlrs_algos::linear::sym_eig::sym_eig;

    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }

    // Just past the threshold, so the operator is the smallest one the Lanczos
    // arm ever actually sees in production.
    let n = DENSE_N + 8;
    let d = 4usize;
    let nev = 4usize;
    let k = 10usize;

    // Deterministic pseudo-random cloud (the SplitMix64 mixer, so the test owns
    // its data and needs no fixture). A generic cloud in 4-D at k=10 gives a
    // connected graph with a well-separated low spectrum, which the eigenvalue
    // assertions below verify rather than assume.
    let mut s: u64 = 0x5EED_1234_ABCD_0001;
    let mut x = vec![0.0f64; n * d];
    for v in x.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        *v = ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }

    let aff = build_affinity(&AffinityKind::NearestNeighbors, &x, n, d, k);
    let op = NormAdj::new(aff, n);

    // --- dense arm ---
    let l = op.dense_laplacian();
    let (w, v) = sym_eig(&l, n);
    // `sym_eig` is DESCENDING, so the r-th SMALLEST eigenvalue is `w[n-1-r]`.
    let small: Vec<f64> = (0..=nev).map(|r| w[n - 1 - r]).collect();
    let mut min_gap = f64::INFINITY;
    for r in 0..nev {
        min_gap = min_gap.min(small[r + 1] - small[r]);
    }
    println!(
        "lanczos_matches_dense: n={n}, smallest {} eigenvalues = {:?}",
        nev + 1,
        small
    );
    assert!(
        min_gap > 1e-4,
        "the test operator's low spectrum is (near-)degenerate (min gap \
         {min_gap:e}) — the eigenvectors would be defined only up to a rotation \
         and this comparison would be vacuous"
    );

    // --- iterative arm ---
    let got = lanczos_largest(&op, nev, 0);
    assert_eq!(got.len(), nev * n, "lanczos_largest returns nev columns of n");

    let mut max_abs = 0.0f64;
    for r in 0..nev {
        let c = n - 1 - r;
        // Each eigenvector is defined up to a global sign; align on the dot
        // product before comparing, as the embedding tests do per column.
        let mut dot = 0.0f64;
        for i in 0..n {
            dot += got[r * n + i] * v[i * n + c];
        }
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            let g = sign * got[r * n + i];
            let e = v[i * n + c];
            assert!(g.is_finite(), "lanczos eigenvector {r} entry {i} is not finite");
            max_abs = max_abs.max((g - e).abs());
        }
    }
    println!(
        "lanczos_matches_dense: max |lanczos - dense| over {nev} eigenvectors = \
         {max_abs:e} (min eigenvalue gap {min_gap:e})"
    );
    assert!(
        max_abs <= 1e-8,
        "the Lanczos arm disagrees with the dense sym_eig on the SAME Laplacian \
         by {max_abs:e} (> 1e-8) — the two solvers must return the same \
         eigenvectors, so this is a solver defect, not a tolerance question"
    );
}

/// SPECTRAL-PERF-CPU: the dense/Lanczos equivalence across the WHOLE range the
/// routing constant could plausibly be set to.
///
/// `lanczos_matches_dense` pins one order just past the threshold. This sweeps
/// `n = 65 … 300` — the band that used to take the dense arm, before it was
/// measured at 4.6x slower than Lanczos already at `n = 120` and `DENSE_N` was
/// lowered to 64. Lowering that constant is only free if the two arms agree
/// everywhere in the band it gave up, so this test is the evidence for the
/// constant's value, not merely a smoke check. The upper end stops at 300
/// because the DENSE arm is `O(n³)` in an unoptimized test build — the same
/// sweep run to `n = 511` agrees to 2.0e-14 but costs ~70 s of the run, and the
/// band above 300 is already covered by the two large sklearn oracles (n=700
/// rbf, n=800 kNN), which exercise the Lanczos arm directly.
///
/// Both arms are driven through the ONE public entry point
/// [`smallest_laplacian_vectors`], with `MLRS_SPECTRAL_DENSE_N` forced via the
/// thread-local `abflag` override. That is deliberate: forcing through the real
/// dispatcher is what makes the test exercise the routing rather than two
/// hand-called solvers, and the thread-local override (rather than
/// `std::env::set_var`) is what keeps a sibling test from silently forcing the
/// same knob and turning this into a comparison of one arm against itself.
#[test]
fn lanczos_matches_dense_across_orders() {
    use mlrs_algos::cluster::spectral_affinity::{build_affinity, AffinityKind};
    use mlrs_algos::cluster::spectral_host::{smallest_laplacian_vectors, NormAdj};

    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        return;
    }

    let mut worst = 0.0f64;
    for &(n, d, nev) in &[
        (65usize, 4usize, 3usize),
        (100, 5, 4),
        (128, 8, 3),
        (200, 8, 3),
        (300, 10, 5),
    ] {
        let mut s: u64 = 0xA5A5_0000_1234_5678 ^ (n as u64);
        let mut x = vec![0.0f64; n * d];
        for v in x.iter_mut() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *v = ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
        let k = (n / 10).max(2);
        let aff = build_affinity(&AffinityKind::NearestNeighbors, &x, n, d, k);
        // A disconnected graph has one zero eigenvalue per component, which makes
        // the kept eigenvectors arbitrary within a degenerate null space and the
        // comparison below vacuous. Assert connectivity rather than hope for it.
        assert_eq!(
            mlrs_algos::cluster::spectral_host::connected_components(&aff, n),
            1,
            "n={n}: the test graph must be connected for an elementwise \
             eigenvector comparison to mean anything"
        );
        let op = NormAdj::new(aff, n);

        let dense = {
            let _g = mlrs_backend::abflag::force("MLRS_SPECTRAL_DENSE_N", "100000");
            smallest_laplacian_vectors(&op, nev, 0)
        };
        let lanczos = {
            let _g = mlrs_backend::abflag::force("MLRS_SPECTRAL_DENSE_N", "0");
            smallest_laplacian_vectors(&op, nev, 0)
        };

        let mut max_abs = 0.0f64;
        for r in 0..nev {
            let (a, b) = (&dense[r * n..(r + 1) * n], &lanczos[r * n..(r + 1) * n]);
            let dot: f64 = a.iter().zip(b.iter()).map(|(p, q)| p * q).sum();
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            for (p, q) in a.iter().zip(b.iter()) {
                max_abs = max_abs.max((p - sign * q).abs());
            }
        }
        println!("lanczos_matches_dense n={n} d={d} nev={nev}: max_abs_err = {max_abs:e}");
        assert!(
            max_abs <= 1e-8,
            "n={n}: the dense and Lanczos arms disagree by {max_abs:e} (> 1e-8) — \
             DENSE_N cannot be lowered past this order"
        );
        worst = worst.max(max_abs);
    }
    println!("lanczos_matches_dense_across_orders: worst = {worst:e}");
}

/// BLDR-01: `SpectralEmbedding::new()` (the single-source defaults) equals
/// `SpectralEmbedding::builder().build()` (the builder defaults re-derived from
/// `new`).
#[test]
fn spectral_embedding_defaults_equal() {
    let from_new = SpectralEmbedding::<f32>::new();
    let from_builder = SpectralEmbedding::<f32>::builder()
        .build::<f32>()
        .expect("default SpectralEmbedding builder build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "SpectralEmbedding::new() must equal SpectralEmbedding::builder().build() (BLDR-01)"
    );
}
