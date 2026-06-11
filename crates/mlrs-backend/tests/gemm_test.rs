//! Plan 02-01 — GEMM (PRIM-01) oracle validation.
//!
//! Exercises the host→device GEMM path for both `f32` and `f64` against a
//! host triple-loop reference and the committed `.npz` convention fixtures:
//!
//!   - `gemm_f32_matches_host_ref`     — seeded random shapes (incl. large-K),
//!     f32 device GEMM vs an f64 host triple-loop reference within `F32_TOL`.
//!   - `gemm_f64_matches_host_ref`     — same, f64 device path, capability-gated
//!     by `skip_f64_with_log`, compared within `F64_TOL`.
//!   - `gemm_transpose_matches_host_ref` — `transa` / `transb` equal the
//!     transposed-operand host reference (D-06: no transpose buffer).
//!   - `gemm_npz_fixture_matches`       — `gemm_{f32,f64}_seed42.npz` `A`/`B`/`C`.
//!
//! The bodies are filled in Task 6; the device launch (`prims::gemm::gemm`)
//! lands in Task 5. Until then these are `#[ignore]`d (they compile against the
//! validated host signature). The f64 cases gate on
//! `capability::skip_f64_with_log` (skip, never fail — Criterion 4 / T-05-04).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// f32-precision near-zero floor for the GEMM oracle comparison, mirroring the
/// `F32_ORACLE_NEAR_ZERO_FLOOR` precedent in `pipeline_test.rs`.
///
/// A GEMM accumulates `k` products, so near-cancellation rows produce genuinely
/// tiny results whose *absolute* error stays far inside `1e-5` (a few ×10⁻⁹)
/// while the *relative* term legitimately exceeds `1e-5` purely from f32
/// rounding (~1 ULP). This floor raises the abs-only fallback to an
/// f32-meaningful magnitude for the f32 GEMM cases ONLY; it never loosens the
/// `1e-5` absolute bound — every element must still pass abs ≤ `1e-5`. The f64
/// cases keep the strict core `assert_close`.
const F32_GEMM_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Element-wise f32 GEMM oracle compare: strict abs-AND-rel per `F32_TOL`,
/// except abs-only (still bounded by `tol.abs` = `1e-5`) when
/// `|expected| < F32_GEMM_NEAR_ZERO_FLOOR`. Panics with diagnostic detail.
fn assert_slice_close_f32_gemm(got: &[f64], expected: &[f64], tol: &Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "f32 gemm oracle length mismatch: got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.abs() < F32_GEMM_NEAR_ZERO_FLOOR {
            let abs_err = (g - e).abs();
            assert!(
                abs_err <= tol.abs,
                "f32 gemm near-zero abs check failed at index {i}: got={g:e}, \
                 expected={e:e}, abs_err={abs_err:e} (tol.abs={:e})",
                tol.abs
            );
        } else {
            assert!(
                is_close(g, e, tol),
                "f32 gemm assert_close failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={:e} (tol.abs={:e}, tol.rel={:e})",
                (g - e).abs(),
                tol.abs,
                tol.rel
            );
        }
    }
}

/// Resolve a workspace-root-relative fixture path (matches `pipeline_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Host triple-loop GEMM reference computed in f64 (the oracle ground truth).
///
/// `a` is `m×k` row-major, `b` is `k×n` row-major; `transa`/`transb` read the
/// transposed operand (i.e. `a` stored as `k×m`, `b` as `n×k`). Result is the
/// `m×n` row-major product.
fn host_gemm_ref(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    transa: bool,
    transb: bool,
) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                // a[i,kk]: row-major (m,k) at i*k+kk; transposed buffer (k,m) at kk*m+i.
                let av = if transa { a[kk * m + i] } else { a[i * k + kk] };
                // b[kk,j]: row-major (k,n) at kk*n+j; transposed buffer (n,k) at j*k+kk.
                let bv = if transb { b[j * k + kk] } else { b[kk * n + j] };
                acc += av * bv;
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Run a device GEMM end-to-end for a single shape and return the f64-promoted
/// result alongside the f64 host reference. Generic over the float element
/// type so the f32 and f64 cases share the exact same device path.
fn run_gemm_case<F>(
    a_host: &[F],
    b_host: &[F],
    m: usize,
    k: usize,
    n: usize,
    transa: bool,
    transb: bool,
) -> Vec<F>
where
    F: cubecl::prelude::Numeric + cubecl::prelude::CubePrimitive + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a_host);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, b_host);

    let c_dev = gemm::<F>(
        &mut pool,
        &a_dev,
        (m, k),
        &b_dev,
        (k, n),
        transa,
        transb,
        None,
    )
    .expect("gemm host API rejects nothing for a valid shape");
    c_dev.to_host_metered(&mut pool)
}

/// f32 GEMM vs an f64 host triple-loop reference over several random shapes,
/// including a large-K case for numerical stability (Pitfall 3).
#[test]
fn gemm_f32_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    // Deterministic host inputs (no Rust RNG into the fixture — these are
    // test-local, not committed). Several shapes incl. a large-K case.
    for &(m, k, n) in &[(5usize, 4usize, 3usize), (8, 8, 8), (3, 128, 4)] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();

        let got = run_gemm_case::<f32>(&a, &b, m, k, n, false, false);
        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        let expected = host_gemm_ref(&a64, &b64, m, k, n, false, false);
        let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
        assert_slice_close_f32_gemm(&got64, &expected, &F32_TOL);
    }
}

/// f64 GEMM vs the f64 host reference, capability-gated (skip-with-log).
#[test]
fn gemm_f64_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("gemm f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for &(m, k, n) in &[(5usize, 4usize, 3usize), (8, 8, 8), (3, 128, 4)] {
        let a: Vec<f64> = (0..m * k).map(|i| ((i % 13) as f64) * 0.1 - 0.6).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i % 11) as f64) * 0.1 - 0.5).collect();

        let got = run_gemm_case::<f64>(&a, &b, m, k, n, false, false);
        let expected = host_gemm_ref(&a, &b, m, k, n, false, false);
        assert_slice_close(&got, &expected, &F64_TOL);
    }
}

/// `transa` / `transb` match the transposed-operand host reference (D-06: the
/// transpose is logical index arithmetic, never a materialized buffer).
#[test]
fn gemm_transpose_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, k, n) = (4usize, 5usize, 3usize);
    // transa: A is stored as k×m and read transposed to m×k.
    let a_t: Vec<f32> = (0..k * m).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 9) as f32) * 0.2 - 0.4).collect();
    let got = run_gemm_case::<f32>(&a_t, &b, m, k, n, true, false);
    let a_t64: Vec<f64> = a_t.iter().map(|&x| x as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
    let expected = host_gemm_ref(&a_t64, &b64, m, k, n, true, false);
    let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
    assert_slice_close_f32_gemm(&got64, &expected, &F32_TOL);

    // transb: B is stored as n×k and read transposed to k×n.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
    let b_t: Vec<f32> = (0..n * k).map(|i| ((i % 9) as f32) * 0.2 - 0.4).collect();
    let got = run_gemm_case::<f32>(&a, &b_t, m, k, n, false, true);
    let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let b_t64: Vec<f64> = b_t.iter().map(|&x| x as f64).collect();
    let expected = host_gemm_ref(&a64, &b_t64, m, k, n, false, true);
    let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
    assert_slice_close_f32_gemm(&got64, &expected, &F32_TOL);
}

/// Device GEMM `C` matches the committed numpy `C = A @ B` convention fixture
/// (f32 always, f64 capability-gated).
#[test]
fn gemm_npz_fixture_matches() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    // Fixture geometry (gen_oracle.py GEMM_M/K/N).
    let (m, k, n) = (5usize, 4usize, 3usize);

    // f32 fixture — always runs.
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case: OracleCase =
        load_npz(fixture("gemm_f32_seed42.npz")).expect("load gemm_f32_seed42.npz");
    let a = case.expect_f32("A");
    let b = case.expect_f32("B");
    let c = case.expect_f32("C");
    let got = run_gemm_case::<f32>(a, b, m, k, n, false, false);
    let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
    let c64: Vec<f64> = c.iter().map(|&x| x as f64).collect();
    assert_slice_close_f32_gemm(&got64, &c64, &F32_TOL);

    // f64 fixture — capability-gated.
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gemm npz f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case: OracleCase =
        load_npz(fixture("gemm_f64_seed42.npz")).expect("load gemm_f64_seed42.npz");
    let a = case.expect_f64("A");
    let b = case.expect_f64("B");
    let c = case.expect_f64("C");
    let got = run_gemm_case::<f64>(a, b, m, k, n, false, false);
    assert_slice_close(&got, c, &F64_TOL);
}
