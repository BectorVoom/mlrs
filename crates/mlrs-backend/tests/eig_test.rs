//! Plan 03-02 — symmetric eig (PRIM-05) oracle + invariant test scaffold (Wave 0).
//!
//! Wave-0 SCAFFOLD for the symmetric eigendecomposition primitive. The device
//! launch (`mlrs_backend::prims::eig::eig`) does NOT exist yet — it lands in plan
//! 03-04, which removes the `#[ignore]` markers and wires the real `eig::<F>(...)`
//! call + the residual invariant. Until then each test asserts the committed
//! `np.linalg.eigh` fixture LOADS with the expected named arrays / shapes
//! (descending-order, D-04) and is `#[ignore]`d where it would need the prim, so
//! the scaffold COMPILES and runs on cpu (and rocm).
//!
//! Test functions (VALIDATION.md Per-Task map):
//!   - `eig_symmetric_f32_fixture`  — f32 vs `np.linalg.eigh` (descending,
//!     reversed — D-04).
//!   - `eig_symmetric_f64_fixture`  — f64, capability-gated (cpu runs, rocm
//!     skips-with-log).
//!   - `eig_residual_invariant`     — reference-free `‖A·v − λ·v‖ < tol` (D-09).
//!   - `eig_clustered_invariant`    — clustered-eigenvalue D-08 case via the
//!     residual invariant (per-vector compare is ill-conditioned when
//!     eigenvalues cluster).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`. Eigenvectors are defined only up to a sign, so plan 03-04
//! sign-aligns with `mlrs_core::sign_flip::align_rows` before fixture compare
//! (D-03).

#![allow(unused_imports)]

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{assert_slice_close, is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Resolve a workspace-root-relative fixture path (matches `gemm_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert a loaded eigh fixture carries `A` (n×n), `w` (n descending), `V` (n×n
/// eigenvector columns) with the expected shapes and descending eigenvalues
/// (D-04). Shared by the scaffold load checks until plan 03-04 wires the compare.
fn assert_eigh_fixture_well_formed(case: &OracleCase, n: usize) {
    let a_shape = case.shape("A").expect("fixture has array 'A'");
    let w_shape = case.shape("w").expect("fixture has array 'w'");
    let v_shape = case.shape("V").expect("fixture has array 'V'");
    assert_eq!(a_shape, &[n as u64, n as u64], "A shape (square symmetric)");
    assert_eq!(w_shape, &[n as u64], "w shape");
    assert_eq!(v_shape, &[n as u64, n as u64], "V shape");
    // Fixture stores eigenvalues REVERSED to descending (D-04) — sanity-check.
    let w = case.expect_f64("w");
    for win in w.windows(2) {
        assert!(win[0] >= win[1] - 1e-9, "w must be descending: {:?}", w);
    }
}

/// f32 symmetric eig vs the committed `np.linalg.eigh` fixture (descending,
/// reversed — D-04). Scaffold: asserts the fixture loads + is well-formed. Plan
/// 03-04 removes the `#[ignore]` and compares `eig::<f32>` (after `align_rows`).
#[test]
#[ignore = "prim lands in plan 03-04 (eig::eig not yet implemented)"]
fn eig_symmetric_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let n = 4usize; // gen_oracle.py EIG_N
    let case = load_npz(fixture("eigh_f32_seed42.npz")).expect("load eigh_f32_seed42.npz");
    assert_eigh_fixture_well_formed(&case, n);
}

/// f64 symmetric eig, capability-gated (cpu runs f64; rocm SKIPS-with-log
/// because the CubeCL HIP backend leaves F64 unregistered — EXPECTED).
#[test]
#[ignore = "prim lands in plan 03-04 (eig::eig not yet implemented)"]
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
    assert_eigh_fixture_well_formed(&case, n);
}

/// Reference-free residual invariant `‖A·v − λ·v‖ < tol` (D-09) — basis-invariant,
/// catches bugs the fixture's sign/order can't. Plan 03-04 forms `A·v` with the
/// Phase-2 `gemm` and asserts the residual per eigenpair.
#[test]
#[ignore = "prim lands in plan 03-04 (eig::eig not yet implemented)"]
fn eig_residual_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let case = load_npz(fixture("eigh_f32_seed42.npz")).expect("load eigh_f32_seed42.npz");
    assert!(!case.expect_f32("A").is_empty(), "A present for residual check");
    assert!(!case.expect_f32("V").is_empty(), "V present for residual check");
}

/// Clustered-eigenvalue D-08 case checked via the residual invariant ONLY
/// (per-vector compare is ill-conditioned when eigenvalues cluster). Plan 03-04
/// drives a clustered fixture through the same residual norm.
#[test]
#[ignore = "prim lands in plan 03-04 (eig::eig not yet implemented)"]
fn eig_clustered_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Scaffold: the symmetric fixture stands in until 03-04 lands a clustered blob.
    let case = load_npz(fixture("eigh_f32_seed42.npz")).expect("load eigh_f32_seed42.npz");
    assert!(!case.expect_f32("w").is_empty(), "w present");
}
