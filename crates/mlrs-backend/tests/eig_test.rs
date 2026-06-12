//! Plan 03-04 — symmetric eig (PRIM-05) oracle + invariant tests.
//!
//! These exercise the two-sided cyclic Jacobi symmetric-eigendecomposition
//! primitive (`mlrs_backend::prims::eig::eig`) on cpu (f32 + f64) and rocm (f32;
//! f64 skip-with-log per the CubeCL-HIP F64 gap, D-07). Two complementary
//! checks:
//!
//!   - **Oracle (fixture) compare** — against the committed
//!     `np.linalg.eigh` `.npz` blobs (eigenvalues REVERSED to DESCENDING — D-04,
//!     so they match the device order directly), after sign-aligning each
//!     eigenvector column with `align_rows` (D-03). Reserved for the
//!     well-conditioned symmetric case (per-vector compare is ill-conditioned on
//!     clustered spectra — Pitfall 3).
//!   - **Reference-free residual invariant** — basis-invariant `‖A·v − λ·v‖ <
//!     tol` (D-09), built with the Phase-2 `gemm()` for `A·v`. This catches bugs
//!     the fixture's sign/order cannot, and carries the clustered D-08 case
//!     (per-vector fixture compare is meaningless when eigenvalues cluster).
//!
//! f64 fixture tests gate on `capability::skip_f64_with_log` (cpu runs f64; rocm
//! skips-with-log — EXPECTED, not a defect). Per AGENTS.md §2, tests live in
//! `tests/`, never as an in-source `#[cfg(test)] mod tests`. Eigenvectors are
//! defined only up to a sign, so the fixture compare sign-aligns columns with
//! `mlrs_core::sign_flip::align_rows` before comparing (D-03).

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// f32 near-zero floor for the eig oracle compare, mirroring the SVD test's
/// `F32_SVD_NEAR_ZERO_FLOOR` precedent (D-10 — strict 1e-5 abs, abs-only
/// fallback below the floor; never pre-loosened).
const F32_EIG_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Resolve a workspace-root-relative fixture path (matches `gemm_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("eig tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("eig tests are f32/f64 only"),
    }
}

fn promote<F: bytemuck::Pod>(v: &[F]) -> Vec<f64> {
    v.iter().map(|&x| host_to_f64(x)).collect()
}

/// Run `eig()` on the device for a row-major `n × n` symmetric host matrix `a`,
/// returning `(w, V)` read back to host (f64-promoted). `w` is length `n`
/// (descending — D-04); `V` is column-major `n × n` (`v[c*n + r] = V[r, c]`).
fn run_eig<F>(a: &[F], n: usize) -> (Vec<f64>, Vec<f64>)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a);
    let (w, v) = eig::<F>(&mut pool, &a_dev, n, None).expect("eig on a valid square shape");
    let w_h = promote(&w.to_host_metered(&mut pool));
    let v_h = promote(&v.to_host_metered(&mut pool));
    (w_h, v_h)
}

/// Frobenius / L2 norm of a flat vector.
fn fro(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Split a column-major `n × n` `V` (device layout `v[c*n + r] = V[r, c]`) into a
/// `Vec<Vec<f64>>` of its COLUMNS (each an eigenvector) for `align_rows`.
fn columns_colmajor(v: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|c| (0..n).map(|r| v[c * n + r]).collect())
        .collect()
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

/// Assert the per-eigenpair residual `‖A·v_i − λ_i·v_i‖ < tol` (D-09),
/// basis-invariant. `a` is the row-major `n × n` symmetric matrix; `(w, v)` are
/// the device outputs (`v` column-major). `A·v_i` is formed with the Phase-2
/// `gemm()` (one GEMM of `A · V`), so this reuses the validated matmul rather
/// than hand-rolling one. `tol` bounds each eigenpair's residual L2 norm.
fn assert_eig_residual<F>(a: &[F], n: usize, w: &[f64], v_colmajor: &[f64], tol: f64, label: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // A·V via the Phase-2 GEMM. A is row-major (n×n). V must be presented
    // row-major (n×n) as B so C[r, j] = Σ_k A[r, k]·V[k, j]. The device `v` is
    // column-major (v[c*n + r] = V[r, c]); a ROW-major read of that same buffer
    // is therefore Vᵀ, so pass transb=true to read it back as V.
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a);
    let v_f: Vec<F> = v_colmajor.iter().map(|&x| from_f64::<F>(x)).collect();
    let v_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &v_f);
    let av = gemm::<F>(&mut pool, &a_dev, (n, n), &v_dev, (n, n), false, true, None)
        .expect("A·V gemm");
    let av_h = promote(&av.to_host(&pool)); // row-major (n×n): av_h[r*n + j] = (A·V)[r, j].

    // For each eigenpair j: residual_j = ‖ (A·V)[:, j] − w[j]·V[:, j] ‖.
    for j in 0..n {
        let mut resid = vec![0.0f64; n];
        for r in 0..n {
            let av_rj = av_h[r * n + j];
            let v_rj = v_colmajor[j * n + r]; // V[r, j] (column-major).
            resid[r] = av_rj - w[j] * v_rj;
        }
        let nrm = fro(&resid);
        assert!(
            nrm < tol,
            "{label}: eig residual ‖A·v−λ·v‖={nrm:e} for eigenpair {j} (λ={:e}) exceeds tol={tol:e}",
            w[j]
        );
    }
}

// ===========================================================================
// Oracle (fixture) tests
// ===========================================================================

/// f32 symmetric eig vs the committed `np.linalg.eigh` fixture (descending,
/// reversed — D-04), sign-aligned per `align_rows` (D-03). Compares `w` directly
/// (both descending) and the sign-aligned eigenvector columns within `F32_TOL`
/// (near-zero floored, D-10).
#[test]
fn eig_symmetric_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let n = 4usize; // gen_oracle.py EIG_N
    let case = load_npz(fixture("eigh_f32_seed42.npz")).expect("load eigh_f32_seed42.npz");
    compare_against_fixture::<f32>(&case, n, &F32_TOL, F32_EIG_NEAR_ZERO_FLOOR);
}

/// f64 symmetric eig, capability-gated (cpu runs f64; rocm SKIPS-with-log
/// because the CubeCL HIP backend leaves F64 unregistered — EXPECTED).
#[test]
fn eig_symmetric_f64_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("eig f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let n = 4usize;
    let case = load_npz(fixture("eigh_f64_seed42.npz")).expect("load eigh_f64_seed42.npz");
    compare_against_fixture::<f64>(&case, n, &F64_TOL, 0.0);
}

/// Shared fixture-compare body: run `eig()` on the fixture `A`, compare `w`
/// directly (both descending — D-04) and the sign-aligned eigenvector columns
/// vs numpy (D-03).
fn compare_against_fixture<F>(case: &OracleCase, n: usize, tol: &Tolerance, floor: f64)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let a_f: Vec<F> = case
        .expect_f64("A")
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect();
    let (w, v) = run_eig::<F>(&a_f, n);

    // w compares directly (both descending — D-04).
    let w_ref: Vec<f64> = case.expect_f64("w").to_vec();
    assert_close_floored(&w, &w_ref, tol, floor, "w");

    // Eigenvector columns: align sign before compare (D-03). The fixture `V`
    // stores eigenvectors as COLUMNS (row-major n×n: V[r, c] at r*n + c); the
    // device `v` is column-major (v[c*n + r] = V[r, c]).
    let v_ref: Vec<f64> = case.expect_f64("V").to_vec();
    let v_ref_cols: Vec<Vec<f64>> = (0..n)
        .map(|c| (0..n).map(|r| v_ref[r * n + c]).collect())
        .collect();

    let v_cols = align_rows(&columns_colmajor(&v, n));
    let v_ref_cols = align_rows(&v_ref_cols);
    for j in 0..n {
        assert_close_floored(&v_cols[j], &v_ref_cols[j], tol, floor, "V col");
    }
}

// ===========================================================================
// Reference-free invariants
// ===========================================================================

/// Reference-free residual invariant `‖A·v − λ·v‖ < tol` (D-09) — basis-
/// invariant, catches bugs the fixture's sign/order can't. Forms `A·v` with the
/// Phase-2 `gemm()` and asserts the residual per eigenpair on the well-
/// conditioned symmetric fixture (f32; runs on cpu AND rocm).
#[test]
fn eig_residual_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let n = 4usize;
    let case = load_npz(fixture("eigh_f32_seed42.npz")).expect("load eigh_f32_seed42.npz");
    let a: Vec<f32> = case.expect_f32("A").to_vec();
    let (w, v) = run_eig::<f32>(&a, n);
    assert_eig_residual::<f32>(&a, n, &w, &v, 1e-4, "symmetric-f32");
}

/// Clustered-eigenvalue D-08 case checked via the residual invariant ONLY
/// (per-vector compare is ill-conditioned when eigenvalues cluster — Pitfall 3).
/// Builds a symmetric matrix with REPEATED eigenvalues (a near-identity block
/// plus one distinct direction) and drives it through the same residual norm
/// (f32; runs on cpu AND rocm).
#[test]
fn eig_clustered_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let n = 4usize;
    // Symmetric matrix with a clustered spectrum: diag(2, 2, 2, 5) rotated into a
    // non-diagonal basis would keep eigenvalues {2,2,2,5}. We instead build a
    // genuinely symmetric matrix with repeated eigenvalues directly: a scaled
    // identity (λ=2, multiplicity 3) plus a rank-1 bump along (1,1,1,1)/2 adding
    // 3 to that direction → eigenvalues {5, 2, 2, 2}, the eigenspace for λ=2 is
    // 3-dimensional (clustered/degenerate — per-vector compare is meaningless).
    let mut a = vec![0.0f32; n * n];
    for r in 0..n {
        a[r * n + r] = 2.0;
    }
    // rank-1 bump: A += 3 * u·uᵀ with u = (1,1,1,1)/2 (‖u‖=1) → adds 3 along u.
    let u = [0.5f32; 4];
    for r in 0..n {
        for c in 0..n {
            a[r * n + c] += 3.0 * u[r] * u[c];
        }
    }
    // A is symmetric by construction (D-06 feeder contract).
    let (w, v) = run_eig::<f32>(&a, n);
    // Basis-invariant residual: ‖A·v − λ·v‖ small for every eigenpair even though
    // three eigenvalues coincide (the eigenvectors within the λ=2 eigenspace are
    // arbitrary, but each must still satisfy the eigen-relation).
    assert_eig_residual::<f32>(&a, n, &w, &v, 1e-3, "clustered-f32");
    // Sanity: the dominant eigenvalue is ~5 (descending — D-04).
    assert!(
        (w[0] - 5.0).abs() < 1e-3,
        "clustered: dominant eigenvalue {} should be ~5 (descending D-04)",
        w[0]
    );
}
