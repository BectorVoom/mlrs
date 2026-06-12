//! Plan 04-02 — Cholesky/SPD-solve primitive (D-02) standalone validation.
//!
//! Exercises the NEW single-cube Cholesky factor + triangular-solve primitive
//! (`mlrs_backend::prims::cholesky::cholesky_solve`) on cpu (f32 + f64) and rocm
//! (f32; f64 skip-with-log per the CubeCL-HIP F64 gap, D-07). Three checks
//! validate the primitive STANDALONE before Ridge (04-05) consumes it, mirroring
//! the Phase-2/3 primitive-first discipline:
//!
//!   - **`‖A·x − b‖` solve invariant** — solve `A·x = b` on the device for the
//!     committed `scipy.linalg.solve(A, b, assume_a="pos")` fixture and assert the
//!     RESIDUAL `‖A·x − b‖` is within 1e-5 (the scale-invariant form of the 1e-5
//!     contract). Also compares `x` directly against the stored scipy reference.
//!   - **`‖L·Lᵀ − A‖` factor invariant** — read back the KERNEL-EMITTED lower
//!     factor `L` (via `cholesky_solve_with_factor`, NOT re-derived on the host),
//!     reconstruct `L·Lᵀ`, and assert it matches the fixture `A` within tolerance.
//!   - **Non-SPD guard** — feed a synthetically indefinite matrix and assert the
//!     host returns `PrimError::NotPositiveDefinite` (negative-pivot flag), never
//!     a NaN-poisoned factor (RESEARCH Pitfall 4).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log — EXPECTED, not a defect, D-07). Per AGENTS.md §2 tests
//! live here, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::{cholesky_solve, cholesky_solve_with_factor};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, PrimError};

/// Cholesky fixture geometry (gen_oracle.py `CHOL_N` × `CHOL_RHS`): A is n×n,
/// b/x are n×rhs, L is n×n.
const CHOL_N: usize = 6;
const CHOL_RHS: usize = 2;

/// Residual / reconstruction tolerance for the well-conditioned SPD fixture. The
/// fixture is `A = MᵀM + nI` (benign condition number), so the f32 device solve
/// reaches ~1e-6; 1e-5 is the project contract.
const SOLVE_TOL: f64 = 1e-5;
/// f64 cpu path is far tighter; keep the same 1e-5 contract bound.
const RECON_TOL: f64 = 1e-5;

/// Resolve a workspace-root-relative fixture path (matches `svd_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the named array exists with exactly `len` elements (flat).
fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cholesky tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("cholesky tests are f32/f64 only"),
    }
}

/// Frobenius norm of a flat matrix.
fn fro(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
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

/// `fixture-dtype` host vector from the f64 fixture array.
fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Shared solve body: load the fixture, run `cholesky_solve` on the device, and
/// assert both `‖A·x − b‖ ≤ 1e-5` and `x` vs the stored scipy reference.
fn check_solve<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load cholesky fixture");
    let a: Vec<F> = fixture_vec::<F>(&case, "A");
    let b: Vec<F> = fixture_vec::<F>(&case, "b");
    let x_ref: Vec<f64> = case.expect_f64("x").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);

    let x_dev = cholesky_solve::<F>(&mut pool, &a_dev, &b_dev, CHOL_N, CHOL_RHS, None)
        .expect("cholesky solve on a valid SPD system");
    let x: Vec<f64> = x_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    x_dev.release_into(&mut pool);

    // (a) ‖A·x − b‖ residual invariant (scale-invariant 1e-5 contract).
    let a64: Vec<f64> = a.iter().map(|&v| host_to_f64(v)).collect();
    let b64: Vec<f64> = b.iter().map(|&v| host_to_f64(v)).collect();
    let ax = matmul(&a64, &x, CHOL_N, CHOL_N, CHOL_RHS);
    let resid: Vec<f64> = ax.iter().zip(b64.iter()).map(|(&p, &q)| p - q).collect();
    let b_fro = fro(&b64).max(1.0);
    let rel = fro(&resid) / b_fro;
    assert!(
        rel <= SOLVE_TOL,
        "‖A·x−b‖/‖b‖={rel:e} exceeds the {SOLVE_TOL:e} solve contract"
    );

    // (b) x vs the scipy reference (the fixture is well-conditioned so a direct
    //     compare holds to 1e-5).
    for (i, (&g, &e)) in x.iter().zip(x_ref.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= SOLVE_TOL + SOLVE_TOL * e.abs(),
            "x[{i}] mismatch vs scipy: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// Shared factor body: read back the KERNEL-EMITTED lower factor `L` and assert
/// `‖L·Lᵀ − A‖` matches the fixture `A` within tolerance (L is NOT re-derived on
/// the host — it is the kernel's `l_out` buffer, the unambiguous L source).
fn check_factor<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load cholesky fixture");
    let a: Vec<F> = fixture_vec::<F>(&case, "A");
    let b: Vec<F> = fixture_vec::<F>(&case, "b");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);

    let (x_dev, l_dev) =
        cholesky_solve_with_factor::<F>(&mut pool, &a_dev, &b_dev, CHOL_N, CHOL_RHS, None)
            .expect("cholesky factor on a valid SPD system");
    // Read back the kernel-written L (row-major n×n, strictly-upper = 0).
    let l: Vec<f64> = l_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    x_dev.release_into(&mut pool);
    l_dev.release_into(&mut pool);

    // Reconstruct L·Lᵀ (n×n) and compare against the fixture A. Lᵀ is L read with
    // transposed indices; build it explicitly then matmul.
    let mut lt = vec![0.0f64; CHOL_N * CHOL_N];
    for i in 0..CHOL_N {
        for j in 0..CHOL_N {
            lt[i * CHOL_N + j] = l[j * CHOL_N + i];
        }
    }
    let llt = matmul(&l, &lt, CHOL_N, CHOL_N, CHOL_N);
    let a64: Vec<f64> = a.iter().map(|&v| host_to_f64(v)).collect();
    let diff: Vec<f64> = llt.iter().zip(a64.iter()).map(|(&p, &q)| p - q).collect();
    let a_fro = fro(&a64).max(1.0);
    let rel = fro(&diff) / a_fro;
    assert!(
        rel <= RECON_TOL,
        "‖L·Lᵀ−A‖/‖A‖={rel:e} exceeds the {RECON_TOL:e} factor contract \
         (L read back from the kernel l_out buffer, not re-derived)"
    );
}

/// LOAD-NOT-JUST-PRESENT check: load the committed `cholesky_f64_seed42.npz` via
/// `mlrs_core::load_npz` and assert the `A`/`b`/`x`/`L` keys exist with the
/// expected n×n / n×rhs shapes. Proves the committed blob is well-formed.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("cholesky_f64_seed42.npz")).expect("load cholesky_f64");
    assert_len(&case, "A", CHOL_N * CHOL_N);
    assert_len(&case, "b", CHOL_N * CHOL_RHS);
    assert_len(&case, "x", CHOL_N * CHOL_RHS);
    assert_len(&case, "L", CHOL_N * CHOL_N);
    assert_eq!(case.shape("A"), Some([CHOL_N as u64, CHOL_N as u64].as_slice()));
    assert_eq!(case.shape("b"), Some([CHOL_N as u64, CHOL_RHS as u64].as_slice()));
}

/// `‖A·x − b‖` solve invariant, f32 (runs on cpu AND rocm).
#[test]
fn cholesky_solves_spd_system_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_solve::<f32>("cholesky_f32_seed42.npz");
}

/// `‖A·x − b‖` solve invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
fn cholesky_solves_spd_system_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_solve::<f64>("cholesky_f64_seed42.npz");
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f32 (runs on cpu AND rocm). Reads the
/// KERNEL-EMITTED L factor (l_out buffer), never re-derives it on the host.
#[test]
fn cholesky_factor_reconstructs_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_factor::<f32>("cholesky_f32_seed42.npz");
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
fn cholesky_factor_reconstructs_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_factor::<f64>("cholesky_f64_seed42.npz");
}

/// Non-SPD guard: feed a synthetically INDEFINITE matrix (a negative diagonal
/// entry makes the leading pivot non-positive) and assert the host returns
/// `PrimError::NotPositiveDefinite` (the negative-pivot flag) rather than a
/// NaN-poisoned factor (RESEARCH Pitfall 4). f32, runs on cpu AND rocm.
#[test]
fn cholesky_rejects_non_spd() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    // A clearly indefinite matrix: a negative diagonal forces a non-positive
    // pivot at index 0 (the very first sqrt argument is negative). Symmetric so
    // it is a legitimate "looks square + symmetric but not SPD" input.
    let n = CHOL_N;
    let rhs = 1usize;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = if i == 0 { -4.0 } else { 2.0 };
    }
    let b = vec![1.0f32; n * rhs];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &b);

    let res = cholesky_solve::<f32>(&mut pool, &a_dev, &b_dev, n, rhs, None);
    match res {
        Err(PrimError::NotPositiveDefinite {
            operand,
            pivot_index,
            pivot_value,
        }) => {
            assert_eq!(operand, "cholesky", "NotPositiveDefinite names the operand");
            assert_eq!(pivot_index, 0, "the negative diagonal is at index 0");
            assert!(
                pivot_value.is_finite() && pivot_value <= 0.0,
                "pivot_value should be the non-positive √ argument, got {pivot_value:e}"
            );
            println!(
                "cholesky non-SPD backend={backend}: rejected at pivot {pivot_index} \
                 (value={pivot_value:e}) — typed error, not a NaN factor"
            );
        }
        Ok(_) => panic!(
            "an indefinite matrix (negative pivot) must return NotPositiveDefinite, \
             not Ok — the SPD guard is broken"
        ),
        Err(other) => panic!("expected NotPositiveDefinite, got a different error: {other:?}"),
    }
}
