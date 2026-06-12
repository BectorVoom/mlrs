//! Plan 04-01 — Cholesky/SPD-solve (D-02) Nyquist scaffold (`#[ignore]` stubs).
//!
//! Wave-0 scaffold for the NEW Cholesky/triangular-solve primitive that Ridge
//! consumes (D-02). The primitive (`mlrs_backend::prims::cholesky`) does not
//! exist yet, so every test here is a COMPILING `#[ignore]` stub that loads its
//! committed fixture and asserts only fixture shape/well-formedness — NO
//! reference to `prims::cholesky` in any compiled body. Plan 04-02 removes the
//! `#[ignore]` markers and wires the real `‖A·x − b‖` / `‖L·Lᵀ − A‖` /
//! `NotPositiveDefinite` assertions against the primitive.
//!
//! This mirrors the Phase-3 Wave-0 pattern (`svd_test.rs` / `eig_test.rs` stubs
//! that asserted fixture load + shape only). f64 functions carry the
//! `skip_f64_with_log` capability gate verbatim (cpu runs f64; rocm skips-with-
//! log — EXPECTED, not a defect, D-07). Per AGENTS.md §2 tests live here, never
//! as an in-source `#[cfg(test)] mod tests`.
//!
//! `fixture_loads` is the load-not-just-present check Task 2's strengthened
//! verify runs with `--ignored`: it loads `cholesky_f64_seed42.npz` via
//! `mlrs_core::load_npz` and asserts keys `A`/`b`/`x`/`L` exist with the
//! expected shapes — proving the committed blob is well-formed.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// Cholesky fixture geometry (gen_oracle.py `CHOL_N` × `CHOL_RHS`): A is n×n,
/// b/x are n×rhs, L is n×n.
const CHOL_N: usize = 6;
const CHOL_RHS: usize = 2;

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

/// LOAD-NOT-JUST-PRESENT check (Task 2's `--ignored` verify target): load the
/// committed `cholesky_f64_seed42.npz` via `mlrs_core::load_npz` and assert the
/// `A`/`b`/`x`/`L` keys exist with the expected n×n / n×rhs shapes. Proves the
/// committed blob is well-formed, not merely present on disk.
#[test]
#[ignore = "04-02 wires the real Cholesky solve; this stub proves the fixture loads"]
fn fixture_loads() {
    let case = load_npz(fixture("cholesky_f64_seed42.npz")).expect("load cholesky_f64");
    // A and L are n×n; b and x are n×rhs.
    assert_len(&case, "A", CHOL_N * CHOL_N);
    assert_len(&case, "b", CHOL_N * CHOL_RHS);
    assert_len(&case, "x", CHOL_N * CHOL_RHS);
    assert_len(&case, "L", CHOL_N * CHOL_N);
    // Shapes are 2-D as written by np.savez.
    assert_eq!(case.shape("A"), Some([CHOL_N as u64, CHOL_N as u64].as_slice()));
    assert_eq!(case.shape("b"), Some([CHOL_N as u64, CHOL_RHS as u64].as_slice()));
}

/// `‖A·x − b‖` invariant, f32 (04-02 wires the real `prims::cholesky` solve).
#[test]
#[ignore = "04-02 removes #[ignore] and wires the real ‖A·x−b‖ solve invariant"]
fn cholesky_solves_spd_system_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("cholesky_f32_seed42.npz")).expect("load cholesky_f32");
    assert_len(&case, "A", CHOL_N * CHOL_N);
    assert_len(&case, "x", CHOL_N * CHOL_RHS);
}

/// `‖A·x − b‖` invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-02 removes #[ignore] and wires the real ‖A·x−b‖ solve invariant"]
fn cholesky_solves_spd_system_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("cholesky_f64_seed42.npz")).expect("load cholesky_f64");
    assert_len(&case, "A", CHOL_N * CHOL_N);
    assert_len(&case, "x", CHOL_N * CHOL_RHS);
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f32 (04-02 wires the factorization).
#[test]
#[ignore = "04-02 removes #[ignore] and wires the real ‖L·Lᵀ−A‖ factor invariant"]
fn cholesky_factor_reconstructs_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("cholesky_f32_seed42.npz")).expect("load cholesky_f32");
    assert_len(&case, "L", CHOL_N * CHOL_N);
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-02 removes #[ignore] and wires the real ‖L·Lᵀ−A‖ factor invariant"]
fn cholesky_factor_reconstructs_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("cholesky_f64_seed42.npz")).expect("load cholesky_f64");
    assert_len(&case, "L", CHOL_N * CHOL_N);
}

/// Non-SPD guard: 04-02 feeds an indefinite matrix and asserts the host returns
/// `PrimError::NotPositiveDefinite` (negative-pivot flag) rather than a NaN
/// factor. The stub only confirms the SPD fixture loads (the indefinite input is
/// constructed in-test by 04-02; no committed non-SPD fixture is needed).
#[test]
#[ignore = "04-02 removes #[ignore] and asserts PrimError::NotPositiveDefinite on a non-SPD input"]
fn cholesky_rejects_non_spd() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("cholesky_f32_seed42.npz")).expect("load cholesky_f32");
    assert_len(&case, "A", CHOL_N * CHOL_N);
}
