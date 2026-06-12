//! Plan 02-04 — covariance / XᵀX (Gram) primitive (PRIM-04) oracle validation.
//!
//! Exercises the device covariance (`prims::covariance::covariance`) for both
//! `f32` and `f64`, both the population (`ddof = 0`) and sample (`ddof = 1`)
//! normalisation conventions, against a DIRECT host reference AND the committed
//! `np.cov` `.npz` convention fixtures:
//!
//!   - `covariance_ddof0_matches` — `covariance(ddof = 0)` (population, `1/n`)
//!     vs a direct f64 host reference AND `cov_ddof0_f64_seed42.npz`.
//!   - `covariance_ddof1_matches` — `covariance(ddof = 1)` (sample, `1/(n−1)`)
//!     vs the host reference AND `cov_ddof1_f64_seed42.npz` (f64) plus
//!     `cov_ddof1_f32_seed42.npz` (f32).
//!
//! The host reference is the DIRECT column-centred `AᵀA / (n − ddof)` computed
//! in f64, and the fixtures are `np.cov(A, rowvar=False, ddof=ddof)` — features
//! are the COLUMNS of `A` (the `(n_samples, n_features)` row-major contract the
//! device API takes). A match validates the device's GEMM(transa)-based Gram +
//! ddof normalisation against an independent oracle, not a tautology.
//!
//! The f64 cases gate on `capability::skip_f64_with_log` (skip, never fail —
//! Criterion 4 / T-05-04). The f32 case uses `F32_TOL` with the near-zero floor
//! precedent from `gemm_test.rs` / `distance_test.rs`: a covariance entry near
//! zero (a feature pair that is nearly uncorrelated after centring) has an f32
//! relative error that can exceed `1e-5` purely from rounding while the absolute
//! error stays far inside it, so below the floor the check falls back to
//! abs-only — it NEVER loosens the `1e-5` absolute bound, and f64 keeps the
//! strict `assert_slice_close`.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::covariance::covariance;
use mlrs_backend::prims::reduce::ReducePath;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// f32-precision near-zero floor for the covariance oracle comparison, mirroring
/// `F32_GEMM_NEAR_ZERO_FLOOR` in `gemm_test.rs` and `F32_DIST_NEAR_ZERO_FLOOR`
/// in `distance_test.rs`. A covariance entry can be genuinely tiny (a nearly
/// uncorrelated feature pair after centring), so its *absolute* error stays far
/// inside `1e-5` while the *relative* term can exceed it purely from f32
/// rounding. This floor raises the abs-only fallback to an f32-meaningful
/// magnitude for the f32 covariance case ONLY; it never loosens the `1e-5`
/// absolute bound. The f64 cases keep the strict comparator.
const F32_COV_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Element-wise f32 covariance oracle compare: strict abs-AND-rel per `F32_TOL`,
/// except abs-only (still bounded by `tol.abs` = `1e-5`) when
/// `|expected| < F32_COV_NEAR_ZERO_FLOOR`. Panics with diagnostic detail.
fn assert_slice_close_f32_cov(got: &[f64], expected: &[f64], tol: &Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "f32 covariance oracle length mismatch: got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.abs() < F32_COV_NEAR_ZERO_FLOOR {
            let abs_err = (g - e).abs();
            assert!(
                abs_err <= tol.abs,
                "f32 covariance near-zero abs check failed at index {i}: got={g:e}, \
                 expected={e:e}, abs_err={abs_err:e} (tol.abs={:e})",
                tol.abs
            );
        } else {
            assert!(
                is_close(g, e, tol),
                "f32 covariance assert_close failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={:e} (tol.abs={:e}, tol.rel={:e})",
                (g - e).abs(),
                tol.abs,
                tol.rel
            );
        }
    }
}

/// Resolve a workspace-root-relative fixture path (matches `gemm_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Direct host covariance reference, computed in f64 (the oracle ground truth —
/// independent of the device's GEMM(transa) Gram).
///
/// `a` is `n_samples × n_features` row-major (observations in rows, features in
/// columns — the `rowvar=False` convention). The result is the `n_features ×
/// n_features` covariance: centre each column by its mean, then
/// `C[p, q] = (Σ_r centred[r, p] · centred[r, q]) / (n_samples − ddof)`.
fn host_cov_ref(a: &[f64], n_samples: usize, n_features: usize, ddof: usize) -> Vec<f64> {
    // Column means.
    let mut mean = vec![0.0f64; n_features];
    for r in 0..n_samples {
        for c in 0..n_features {
            mean[c] += a[r * n_features + c];
        }
    }
    for c in 0..n_features {
        mean[c] /= n_samples as f64;
    }
    // Centred matrix.
    let mut centred = vec![0.0f64; n_samples * n_features];
    for r in 0..n_samples {
        for c in 0..n_features {
            centred[r * n_features + c] = a[r * n_features + c] - mean[c];
        }
    }
    // Gram / (n − ddof).
    let denom = (n_samples - ddof) as f64;
    let mut cov = vec![0.0f64; n_features * n_features];
    for p in 0..n_features {
        for q in 0..n_features {
            let mut acc = 0.0f64;
            for r in 0..n_samples {
                acc += centred[r * n_features + p] * centred[r * n_features + q];
            }
            cov[p * n_features + q] = acc / denom;
        }
    }
    cov
}

/// Run a device covariance end-to-end for a single matrix and return the result.
/// Generic over the float element type so the f32 and f64 cases share the exact
/// same device path.
fn run_covariance_case<F>(
    a_host: &[F],
    n_samples: usize,
    n_features: usize,
    ddof: u32,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a_host);

    let cov_dev = covariance::<F>(
        &mut pool,
        &a_dev,
        (n_samples, n_features),
        ddof,
        None,
        // Shared path is always portable; the reduction's plane path is gated
        // separately and validated in reduce_test.rs.
        ReducePath::Shared,
    )
    .expect("covariance host API rejects nothing for a valid shape");
    cov_dev.to_host_metered(&mut pool)
}

/// Population covariance `ddof = 0` (`1/n`) vs the direct f64 host reference
/// over several shapes AND the `cov_ddof0_f64_seed42.npz` fixture.
#[test]
fn covariance_ddof0_matches() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("covariance ddof0 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    // Host-reference sweep (f64), several shapes incl. a wider feature dim.
    for &(ns, nf) in &[(7usize, 4usize), (5, 5), (12, 3)] {
        let a: Vec<f64> = (0..ns * nf).map(|i| ((i % 13) as f64) * 0.1 - 0.6).collect();
        let got = run_covariance_case::<f64>(&a, ns, nf, 0);
        let expected = host_cov_ref(&a, ns, nf, 0);
        assert_slice_close(&got, &expected, &F64_TOL);
    }

    // np.cov(A, rowvar=False, ddof=0) fixture (population).
    let case: OracleCase =
        load_npz(fixture("cov_ddof0_f64_seed42.npz")).expect("load cov_ddof0_f64_seed42.npz");
    let a = case.expect_f64("A");
    let c = case.expect_f64("C");
    // Fixture geometry: COV_N_SAMPLES × COV_N_FEATURES (gen_oracle.py).
    let (ns, nf) = (7usize, 4usize);
    let got = run_covariance_case::<f64>(a, ns, nf, 0);
    assert_slice_close(&got, c, &F64_TOL);

    println!(
        "covariance ddof0 f64 backend={backend}: population covariance matches host ref + np.cov fixture"
    );
}

/// Sample covariance `ddof = 1` (`1/(n−1)`) vs the direct f64 host reference AND
/// the `cov_ddof1_f64_seed42.npz` (f64) + `cov_ddof1_f32_seed42.npz` (f32)
/// fixtures. The f64 arm is capability-gated; the f32 arm always runs.
#[test]
fn covariance_ddof1_matches() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    // Fixture geometry: COV_N_SAMPLES × COV_N_FEATURES (gen_oracle.py).
    let (ns, nf) = (7usize, 4usize);

    // --- f32 sample covariance — always runs. ---
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");
    let case: OracleCase =
        load_npz(fixture("cov_ddof1_f32_seed42.npz")).expect("load cov_ddof1_f32_seed42.npz");
    let a32 = case.expect_f32("A");
    let c32 = case.expect_f32("C");
    let got32 = run_covariance_case::<f32>(a32, ns, nf, 1);
    let got64: Vec<f64> = got32.iter().map(|&v| v as f64).collect();
    let c64: Vec<f64> = c32.iter().map(|&v| v as f64).collect();
    assert_slice_close_f32_cov(&got64, &c64, &F32_TOL);
    // f32 device covariance vs the f64 host reference (the direct, independent
    // centred-AᵀA / (n−1) oracle), to confirm it is not just matching np.cov.
    let a32_64: Vec<f64> = a32.iter().map(|&v| v as f64).collect();
    let expected_host = host_cov_ref(&a32_64, ns, nf, 1);
    assert_slice_close_f32_cov(&got64, &expected_host, &F32_TOL);

    // --- f64 sample covariance — capability-gated. ---
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("covariance ddof1 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        println!("covariance ddof1 f32 backend={backend}: sample covariance matches host ref + np.cov fixture");
        return;
    }

    // Host-reference sweep (f64), several shapes.
    for &(s, f) in &[(7usize, 4usize), (5, 5), (12, 3)] {
        let a: Vec<f64> = (0..s * f).map(|i| ((i % 13) as f64) * 0.1 - 0.6).collect();
        let got = run_covariance_case::<f64>(&a, s, f, 1);
        let expected = host_cov_ref(&a, s, f, 1);
        assert_slice_close(&got, &expected, &F64_TOL);
    }

    // np.cov(A, rowvar=False, ddof=1) fixture (sample).
    let case: OracleCase =
        load_npz(fixture("cov_ddof1_f64_seed42.npz")).expect("load cov_ddof1_f64_seed42.npz");
    let a = case.expect_f64("A");
    let c = case.expect_f64("C");
    let got = run_covariance_case::<f64>(a, ns, nf, 1);
    assert_slice_close(&got, c, &F64_TOL);

    println!(
        "covariance ddof1 backend={backend}: sample covariance matches host ref + np.cov fixture (f32 + f64)"
    );
}
