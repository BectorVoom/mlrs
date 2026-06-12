//! Plan 03-02 — SVD (PRIM-05) oracle + invariant test scaffold (Nyquist Wave 0).
//!
//! These are the Wave-0 SCAFFOLD tests for the thin SVD primitive. The device
//! launch (`mlrs_backend::prims::svd::svd`) does NOT exist yet — it lands in
//! plan 03-03, which removes the `#[ignore]` markers and wires the real
//! `svd::<F>(...)` call + the algebraic invariant assertions. Until then each
//! test:
//!
//!   - asserts the committed numpy `.npz` fixture LOADS and carries the expected
//!     named arrays / shapes (proves the fixture + `fixture()` resolver are
//!     wired), and
//!   - is `#[ignore]`d where it would otherwise need the not-yet-existing prim,
//!     so the scaffold COMPILES and runs on cpu (and rocm) before any kernel
//!     exists (Nyquist: every behavior has a failing/ignored test to drive it).
//!
//! Test functions (VALIDATION.md Per-Task map):
//!   - `svd_tall_f32_fixture`          — tall (m≥n) f32 vs `np.linalg.svd`.
//!   - `svd_tall_f64_fixture`          — tall f64, capability-gated (cpu runs,
//!     rocm skips-with-log — CubeCL HIP has F64 unregistered).
//!   - `svd_wide_f32_fixture`          — wide (m<n) f32, the Aᵀ-swap path (D-05).
//!   - `svd_reconstruction_invariant`  — `‖U·diag(S)·Vᵀ − A‖ < tol`.
//!   - `svd_orthonormality_invariant`  — `‖UᵀU − I‖` / `‖VᵀV − I‖ < tol`.
//!   - `svd_degenerate_invariants`     — rank-deficient / repeated / near-identity
//!     via invariants only (D-08).
//!   - `svd_moderate_256x64`           — moderate ~256×64 case exercising the
//!     convergence loop on rocm (D-08).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`. Singular vectors are only defined up to a sign, so plans 03/04
//! sign-align with `mlrs_core::sign_flip::align_rows` before fixture compare
//! (D-03); the import is referenced here so the scaffold matches the analog.

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

/// Assert a loaded SVD fixture carries the named thin-SVD arrays with the
/// expected `(m, k)` / `(k,)` / `(k, n)` shapes (`k = min(m, n)`). Shared by the
/// fixture-load scaffold checks until plan 03-03 wires the device compare.
fn assert_svd_fixture_well_formed(case: &OracleCase, m: usize, n: usize) {
    let k = m.min(n);
    let u_shape = case.shape("U").expect("fixture has array 'U'");
    let s_shape = case.shape("S").expect("fixture has array 'S'");
    let vt_shape = case.shape("Vt").expect("fixture has array 'Vt'");
    let a_shape = case.shape("A").expect("fixture has array 'A'");
    assert_eq!(a_shape, &[m as u64, n as u64], "A shape");
    assert_eq!(u_shape, &[m as u64, k as u64], "U shape (thin SVD)");
    assert_eq!(s_shape, &[k as u64], "S shape");
    assert_eq!(vt_shape, &[k as u64, n as u64], "Vt shape (thin SVD)");
    // np.linalg.svd returns S descending (D-04) — sanity-check monotonicity.
    let s = case.expect_f64("S");
    for w in s.windows(2) {
        assert!(w[0] >= w[1] - 1e-9, "S must be descending: {:?}", s);
    }
}

/// Tall (m≥n) f32 SVD vs the committed `np.linalg.svd` fixture (D-04/D-09).
///
/// Scaffold: asserts the fixture loads + is well-formed. Plan 03-03 removes the
/// `#[ignore]` and compares the device `svd::<f32>` output (after `align_rows`).
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_tall_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, n) = (8usize, 4usize); // gen_oracle.py SVD_TALL
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32_seed42.npz");
    assert_svd_fixture_well_formed(&case, m, n);
}

/// Tall f64 SVD, capability-gated (cpu runs f64; rocm SKIPS-with-log because the
/// CubeCL HIP backend leaves F64 unregistered — EXPECTED, not a defect).
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_tall_f64_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("svd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let (m, n) = (8usize, 4usize);
    let case = load_npz(fixture("svd_tall_f64_seed42.npz")).expect("load svd_tall_f64_seed42.npz");
    assert_svd_fixture_well_formed(&case, m, n);
}

/// Wide (m<n) f32 SVD — exercises the Aᵀ-swap path (run Jacobi on Aᵀ, swap
/// U↔V; D-05) so the primitive is shape-agnostic for PCA/TruncatedSVD.
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_wide_f32_fixture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let (m, n) = (4usize, 8usize); // gen_oracle.py SVD_WIDE
    let case = load_npz(fixture("svd_wide_f32_seed42.npz")).expect("load svd_wide_f32_seed42.npz");
    assert_svd_fixture_well_formed(&case, m, n);
}

/// Reference-free reconstruction invariant `‖U·diag(S)·Vᵀ − A‖ < tol` (D-09).
/// Basis-invariant, so it catches bugs the fixture's sign/order can't. Plan
/// 03-03 builds the product with the Phase-2 `gemm` and asserts the norm.
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_reconstruction_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Scaffold: confirm the fixture A is available to reconstruct against.
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32_seed42.npz");
    assert!(!case.expect_f32("A").is_empty(), "A present for reconstruction");
}

/// Reference-free orthonormality invariant `‖UᵀU − I‖` / `‖VᵀV − I‖ < tol`
/// (D-09). Plan 03-03 forms the Gram products and asserts deviation from I.
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_orthonormality_invariant() {
    let _ = env_logger::builder().is_test(true).try_init();
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32_seed42.npz");
    assert!(!case.expect_f32("U").is_empty(), "U present for orthonormality");
}

/// Degenerate D-08 cases (rank-deficient / repeated / near-identity) checked via
/// the basis-invariant reconstruction + orthonormality norms ONLY — per-vector
/// fixture compare is ill-conditioned when singular values repeat. Plan 03-03
/// drives a degenerate fixture through the same invariants.
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_degenerate_invariants() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Scaffold: the tall fixture stands in until 03-03 lands a degenerate blob.
    let case = load_npz(fixture("svd_tall_f32_seed42.npz")).expect("load svd_tall_f32_seed42.npz");
    assert!(!case.expect_f32("S").is_empty(), "S present");
}

/// Moderate ~256×64 case that exercises the Jacobi convergence loop and reduction
/// reuse on the rocm GPU beyond toy sizes (D-08). Plan 03-03 generates the input
/// in-test (no committed fixture — too large) and checks the invariants.
#[test]
#[ignore = "prim lands in plan 03-03 (svd::svd not yet implemented)"]
fn svd_moderate_256x64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    // Scaffold placeholder: the device path is required, so this stays ignored
    // until 03-03 implements svd::svd. Asserting the runtime client resolves is a
    // cheap smoke that the backend is wired.
    let _client = runtime::active_client();
}
