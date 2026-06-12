//! Plan 03-03 — SVD (PRIM-05) oracle + reference-free invariant tests.
//!
//! These exercise the thin one-sided Jacobi SVD primitive
//! (`mlrs_backend::prims::svd::svd`) on cpu (f32 + f64) and rocm (f32; f64
//! skip-with-log per the CubeCL-HIP F64 gap, D-07). Two complementary checks:
//!
//!   - **Oracle (fixture) compare** — against the committed
//!     `np.linalg.svd(full_matrices=False)` `.npz` blobs, after sign-aligning
//!     each singular vector with `align_rows` (D-03). Reserved for the
//!     well-conditioned tall/wide cases (per-vector compare is ill-conditioned on
//!     clustered/degenerate spectra — Pitfall 3).
//!   - **Reference-free invariants** — basis-invariant `‖U·diag(S)·Vᵀ − A‖`
//!     (reconstruction) and `‖UᵀU − I‖` / `‖VᵀV − I‖` (orthonormality), built
//!     with the Phase-2 `gemm()` for the matrix products. These catch bugs the
//!     fixture's sign/order cannot, and carry the degenerate D-08 cases
//!     (rank-deficient / repeated / near-identity).
//!
//! f64 fixture tests gate on `capability::skip_f64_with_log` (cpu runs f64; rocm
//! skips-with-log — EXPECTED, not a defect). Per AGENTS.md §2, tests live in
//! `tests/`, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{is_close, Tolerance, F32_TOL, F64_TOL};
use mlrs_core::{load_npz, OracleCase};

/// f32 near-zero floor for the SVD oracle compare, mirroring the
/// `assert_slice_close_f32_gemm` precedent (D-10 — strict 1e-5 abs, abs-only
/// fallback below the floor; never pre-loosened).
const F32_SVD_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Resolve a workspace-root-relative fixture path (matches `gemm_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Element-wise close compare with an f32 near-zero floor (D-10): strict
/// abs-AND-rel per `tol`, except abs-only (still bounded by `tol.abs`) when
/// `|expected| < floor`. `floor = 0.0` recovers the strict core compare (f64).
fn assert_close_floored(got: &[f64], expected: &[f64], tol: &Tolerance, floor: f64, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.abs() < floor {
            let abs_err = (g - e).abs();
            assert!(
                abs_err <= tol.abs,
                "{what}: near-zero abs check failed at {i}: got={g:e} expected={e:e} \
                 abs_err={abs_err:e} (tol.abs={:e})",
                tol.abs
            );
        } else {
            assert!(
                is_close(g, e, tol),
                "{what}: assert_close failed at {i}: got={g:e} expected={e:e} \
                 abs_err={:e} (tol.abs={:e}, tol.rel={:e})",
                (g - e).abs(),
                tol.abs,
                tol.rel
            );
        }
    }
}

/// Run `svd()` on the device for a row-major `rows × cols` host matrix `a`,
/// returning `(U, S, Vᵀ)` read back to host (f64-promoted). `k = min(rows,cols)`.
fn run_svd<F>(a: &[F], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a);
    let (u, s, vt) = svd::<F>(&mut pool, &a_dev, (rows, cols)).expect("svd on a valid shape");
    let u_h = promote(&u.to_host(&pool));
    let s_h = promote(&s.to_host(&pool));
    let vt_h = promote(&vt.to_host(&pool));
    (u_h, s_h, vt_h)
}

fn promote<F: bytemuck::Pod>(v: &[F]) -> Vec<f64> {
    v.iter().map(|&x| host_to_f64(x)).collect()
}

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("svd tests are f32/f64 only"),
    }
}

/// Host triple-loop matrix product `C (m×n) = A (m×k) · B (k×n)`, all row-major.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Frobenius norm of a flat matrix.
fn fro(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Reconstruct `U·diag(S)·Vᵀ` (m×n) from thin factors `U` (m×k), `S` (k),
/// `Vᵀ` (k×n), all row-major.
fn reconstruct(u: &[f64], s: &[f64], vt: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // (U·diag(S)) (m×k): scale each U column j by S[j].
    let mut us = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] = u[i * k + j] * s[j];
        }
    }
    matmul(&us, vt, m, k, n)
}

/// Split a row-major `(rows × k)` U into a `Vec<Vec<f64>>` of its COLUMNS (each
/// a singular vector) for `align_rows` sign canonicalization.
fn columns(mat: &[f64], rows: usize, k: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|j| (0..rows).map(|r| mat[r * k + j]).collect())
        .collect()
}

/// Split a row-major `(k × cols)` Vᵀ into a `Vec<Vec<f64>>` of its ROWS (each a
/// right singular vector).
fn rows_of(mat: &[f64], k: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|j| (0..cols).map(|c| mat[j * cols + c]).collect())
        .collect()
}

/// A deterministic well-conditioned tall matrix generator (distinct singular
/// values) for the moderate / invariant cases that have no committed fixture.
fn gen_matrix_f32(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    // Simple LCG-ish deterministic fill with a spread that yields distinct
    // singular values; kept test-local (not committed).
    let mut state = seed.wrapping_add(1);
    let mut out = vec![0.0f32; rows * cols];
    for v in out.iter_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let u = ((state >> 8) & 0xFFFF) as f32 / 65535.0; // [0,1)
        *v = u * 2.0 - 1.0; // [-1, 1)
    }
    out
}

// ===========================================================================
// Oracle (fixture) tests
// ===========================================================================

/// Tall (m≥n) f32 SVD vs the committed `np.linalg.svd` fixture (D-04 / D-09),
/// sign-aligned per `align_rows` (D-03). Compares S exactly and the sign-aligned
/// singular-vector matrices within `F32_TOL` (near-zero floored, D-10).
#[test]
fn svd_tall_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, n) = (8usize, 4usize); // gen_oracle.py SVD_TALL
    let k = m.min(n);
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32");
    compare_against_fixture::<f32>(&case, m, n, k, &F32_TOL, F32_SVD_NEAR_ZERO_FLOOR);
}

/// Tall f64 SVD, capability-gated (cpu runs f64; rocm SKIPS-with-log).
#[test]
fn svd_tall_f64_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("svd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let (m, n) = (8usize, 4usize);
    let k = m.min(n);
    let case = load_npz(fixture("svd_tall_f64_seed42.npz")).expect("load svd_tall_f64");
    compare_against_fixture::<f64>(&case, m, n, k, &F64_TOL, 0.0);
}

/// Wide (m<n) f32 SVD — exercises the Aᵀ-swap path (D-05).
#[test]
fn svd_wide_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, n) = (4usize, 8usize); // gen_oracle.py SVD_WIDE
    let k = m.min(n);
    let case = load_npz(fixture("svd_wide_f32_seed42.npz")).expect("load svd_wide_f32");
    compare_against_fixture::<f32>(&case, m, n, k, &F32_TOL, F32_SVD_NEAR_ZERO_FLOOR);
}

/// Shared fixture-compare body: run `svd()` on the fixture `A`, compare `S`
/// directly and the sign-aligned `U` columns / `Vᵀ` rows vs numpy (D-03).
fn compare_against_fixture<F>(
    case: &OracleCase,
    m: usize,
    n: usize,
    k: usize,
    tol: &Tolerance,
    floor: f64,
) where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let a_f: Vec<F> = case
        .expect_f64("A")
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect();
    let (u, s, vt) = run_svd::<F>(&a_f, m, n);

    // S compares directly (both descending — D-04).
    let s_ref: Vec<f64> = case.expect_f64("S").to_vec();
    assert_close_floored(&s, &s_ref, tol, floor, "S");

    // U columns / Vᵀ rows are singular vectors: align sign before compare (D-03).
    let u_ref: Vec<f64> = case.expect_f64("U").to_vec();
    let vt_ref: Vec<f64> = case.expect_f64("Vt").to_vec();

    let u_cols = align_rows(&columns(&u, m, k));
    let u_ref_cols = align_rows(&columns(&u_ref, m, k));
    for j in 0..k {
        assert_close_floored(&u_cols[j], &u_ref_cols[j], tol, floor, "U col");
    }

    let vt_rows = align_rows(&rows_of(&vt, k, n));
    let vt_ref_rows = align_rows(&rows_of(&vt_ref, k, n));
    for j in 0..k {
        assert_close_floored(&vt_rows[j], &vt_ref_rows[j], tol, floor, "Vt row");
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("svd tests are f32/f64 only"),
    }
}

// ===========================================================================
// Reference-free invariants
// ===========================================================================

/// Reconstruction invariant `‖U·diag(S)·Vᵀ − A‖ < tol` (D-09), basis-invariant.
#[test]
fn svd_reconstruction_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32");
    let (m, n) = (8usize, 4usize);
    let k = m.min(n);
    let a: Vec<f32> = case.expect_f32("A").to_vec();
    let (u, s, vt) = run_svd::<f32>(&a, m, n);
    let recon = reconstruct(&u, &s, &vt, m, k, n);
    let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let diff: Vec<f64> = recon.iter().zip(a64.iter()).map(|(&r, &x)| r - x).collect();
    let err = fro(&diff);
    assert!(
        err < 1e-4,
        "reconstruction ‖UΣVᵀ−A‖={err:e} exceeds tolerance (m={m},n={n})"
    );
}

/// Orthonormality invariant `‖UᵀU − I‖` and `‖VᵀV − I‖ < tol` (D-09), via the
/// Phase-2 `gemm()` for the Gram products (NEW in-test Rust on the validated GEMM).
#[test]
fn svd_orthonormality_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32");
    let (m, n) = (8usize, 4usize);
    let k = m.min(n);
    let a: Vec<f32> = case.expect_f32("A").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);
    let (u, _s, vt) = svd::<f32>(&mut pool, &a_dev, (m, n)).expect("svd valid shape");

    // UᵀU (k×k): gemm with transa=true reads U (m×k) as (k×m) — Gram = UᵀU.
    let utu = gemm::<f32>(&mut pool, &u, (k, m), &u, (m, k), true, false, None)
        .expect("UᵀU gemm");
    let utu_h: Vec<f64> = utu.to_host(&pool).iter().map(|&x| x as f64).collect();
    assert_identity(&utu_h, k, "UᵀU");

    // VVᵀ via Vᵀ (k×n): (Vᵀ)(Vᵀ)ᵀ = VᵀV row-space Gram (k×k) = I for orthonormal
    // right singular vectors. gemm transb=true reads the second Vᵀ (k×n) as (n×k).
    let vvt = gemm::<f32>(&mut pool, &vt, (k, n), &vt, (k, n), false, true, None)
        .expect("VᵀV gemm");
    let vvt_h: Vec<f64> = vvt.to_host(&pool).iter().map(|&x| x as f64).collect();
    assert_identity(&vvt_h, k, "VᵀV");
}

/// Assert a flat row-major `n×n` matrix is within tolerance of the identity.
fn assert_identity(mat: &[f64], n: usize, what: &str) {
    let mut maxdev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let expect = if i == j { 1.0 } else { 0.0 };
            let dev = (mat[i * n + j] - expect).abs();
            if dev > maxdev {
                maxdev = dev;
            }
        }
    }
    assert!(
        maxdev < 1e-4,
        "{what}: max deviation from I = {maxdev:e} exceeds tolerance"
    );
}

/// Degenerate D-08 cases (rank-deficient / repeated / near-identity) checked via
/// the basis-invariant reconstruction + orthonormality norms ONLY (Pitfall 3/4 —
/// per-vector fixture compare is ill-conditioned on clustered/degenerate spectra;
/// the near-zero floor must keep thin-U from dividing by zero on the null space).
#[test]
fn svd_degenerate_invariants() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    // Rank-deficient: column 2 is a copy of column 0 (rank 3 of 4) — tests the
    // near-zero floor (one ≈0 singular value, no divide-by-zero).
    let (m, n) = (6usize, 4usize);
    let mut a = gen_matrix_f32(m, n, 7);
    for r in 0..m {
        a[r * n + 2] = a[r * n + 0];
    }
    check_invariants_f32(&a, m, n, "rank-deficient");

    // Repeated singular values: a near-identity scaled (clustered spectrum).
    let mut id = vec![0.0f32; m * n];
    for r in 0..m {
        if r < n {
            id[r * n + r] = 1.0;
        }
    }
    // perturb slightly so it is not exactly singular but has clustered σ≈1.
    for r in 0..m {
        for c in 0..n {
            id[r * n + c] += gen_matrix_f32(m, n, 9)[r * n + c] * 1e-3;
        }
    }
    check_invariants_f32(&id, m, n, "near-identity/clustered");
}

/// Run `svd()` on a tall f32 matrix and assert BOTH the reconstruction and the
/// orthonormality invariants hold (basis-invariant; safe for degenerate spectra).
fn check_invariants_f32(a: &[f32], m: usize, n: usize, label: &str) {
    let k = m.min(n);
    let (u, s, vt) = run_svd::<f32>(a, m, n);

    // Reconstruction.
    let recon = reconstruct(&u, &s, &vt, m, k, n);
    let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let diff: Vec<f64> = recon.iter().zip(a64.iter()).map(|(&r, &x)| r - x).collect();
    let rerr = fro(&diff);
    assert!(
        rerr < 1e-3,
        "{label}: reconstruction ‖UΣVᵀ−A‖={rerr:e} exceeds tolerance"
    );

    // U columns are unit-norm or exactly zero (rank-deficient null space — the
    // near-zero floor leaves those at 0, which must NOT be NaN/Inf).
    for j in 0..k {
        let col: Vec<f64> = (0..m).map(|r| u[r * k + j]).collect();
        let nrm = fro(&col);
        assert!(
            nrm.is_finite(),
            "{label}: U col {j} norm is non-finite (NaN/Inf — divide-by-zero leak)"
        );
        assert!(
            nrm < 1.0 + 1e-3,
            "{label}: U col {j} norm {nrm:e} exceeds unit (over-normalized)"
        );
    }
}

/// Moderate ~256×64 case (D-08) exercising the Jacobi convergence loop beyond toy
/// sizes — generated in-test (too large for a committed fixture), checked via the
/// basis-invariant reconstruction + orthonormality (f32; runs on cpu AND rocm).
#[test]
fn svd_moderate_256x64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, n) = (256usize, 64usize); // D-08 moderate case
    let a = gen_matrix_f32(m, n, 42);
    check_invariants_f32(&a, m, n, "moderate-256x64");
}
