//! Plan 04-01 — TruncatedSVD (DECOMP-02) Nyquist scaffold (`#[ignore]` stubs).
//!
//! Wave-0 scaffold for the `TruncatedSVD` estimator = thin SVD of UNCENTERED X
//! (D-01/D-02). The estimator does not exist yet, so every test here is a
//! COMPILING `#[ignore]` stub that loads its committed sklearn fixture and
//! asserts only fixture shape/well-formedness — NO reference to the
//! `TruncatedSVD` symbol in any compiled body. The fixture is generated with the
//! DETERMINISTIC `algorithm='arpack'` (NOT randomized, D-07). Plan 04-04 removes
//! the `#[ignore]` markers and wires the real `components_`/`explained_variance_`
//! (= variance of transformed columns, NOT S²/(n−1) — RESEARCH Pitfall 2)/
//! `singular_values_`/`transform` vs sklearn comparison after `align_rows`.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu
//! runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// TruncatedSVD geometry (gen_oracle.py `TSVD_SHAPE` = 10×5,
/// `TSVD_N_COMPONENTS` = 3).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 5;
const N_COMPONENTS: usize = 3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

/// `components_`/`singular_values_` vs sklearn arpack after `align_rows`, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires components_/singular_values_ vs sklearn arpack (align_rows)"]
fn truncated_svd_components_singular_values_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    assert_len(&case, "X", N_SAMPLES * N_FEATURES);
    assert_len(&case, "components_", N_COMPONENTS * N_FEATURES);
    assert_len(&case, "singular_values_", N_COMPONENTS);
}

/// `components_`/`singular_values_` vs sklearn arpack, f64 (cpu runs; rocm skips).
#[test]
#[ignore = "04-04 removes #[ignore] and wires components_/singular_values_ vs sklearn arpack (align_rows)"]
fn truncated_svd_components_singular_values_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    assert_len(&case, "components_", N_COMPONENTS * N_FEATURES);
    assert_len(&case, "singular_values_", N_COMPONENTS);
}

/// `explained_variance_` (= var of transformed columns, Pitfall 2) vs sklearn, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires explained_variance_ (var of transform cols) vs sklearn"]
fn truncated_svd_explained_variance_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    assert_len(&case, "explained_variance_", N_COMPONENTS);
}

/// `explained_variance_` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-04 removes #[ignore] and wires explained_variance_ (var of transform cols) vs sklearn"]
fn truncated_svd_explained_variance_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd ev f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    assert_len(&case, "explained_variance_", N_COMPONENTS);
}

/// `transform(X)` vs sklearn arpack after `align_rows`, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires transform(X) vs sklearn arpack (align_rows)"]
fn truncated_svd_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("truncated_svd_f32_seed42.npz")).expect("load tsvd_f32");
    assert_len(&case, "transform", N_SAMPLES * N_COMPONENTS);
}

/// `transform(X)` vs sklearn arpack, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-04 removes #[ignore] and wires transform(X) vs sklearn arpack (align_rows)"]
fn truncated_svd_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("tsvd transform f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("truncated_svd_f64_seed42.npz")).expect("load tsvd_f64");
    assert_len(&case, "transform", N_SAMPLES * N_COMPONENTS);
}
