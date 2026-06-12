//! Plan 04-01 — PCA (DECOMP-01) Nyquist scaffold (`#[ignore]` stubs).
//!
//! Wave-0 scaffold for the `PCA` estimator = SVD of CENTERED X (D-01, NOT
//! eig-of-covariance). The estimator does not exist yet, so every test here is a
//! COMPILING `#[ignore]` stub that loads its committed sklearn
//! `svd_solver='full'` fixture and asserts only fixture shape/well-formedness —
//! NO reference to the `PCA` symbol in any compiled body. Plan 04-04 removes the
//! `#[ignore]` markers and wires the real `components_`/`explained_variance_`/
//! `explained_variance_ratio_`/`singular_values_`/`mean_`/`transform`/
//! `inverse_transform` vs sklearn comparison after `align_rows` sign alignment
//! (D-03, `svd_flip(u_based_decision=False)`).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu
//! runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// PCA tall-case geometry (gen_oracle.py `PCA_TALL` = 10×4,
/// `PCA_N_COMPONENTS_TALL` = 3).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 4;
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

/// `components_`/`mean_`/`singular_values_` vs sklearn after `align_rows`, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires components_/mean_/singular_values_ vs sklearn (align_rows)"]
fn pca_components_mean_singular_values_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    assert_len(&case, "X", N_SAMPLES * N_FEATURES);
    assert_len(&case, "components_", N_COMPONENTS * N_FEATURES);
    assert_len(&case, "mean_", N_FEATURES);
    assert_len(&case, "singular_values_", N_COMPONENTS);
}

/// `components_`/`mean_`/`singular_values_` vs sklearn, f64 (cpu runs; rocm skips).
#[test]
#[ignore = "04-04 removes #[ignore] and wires components_/mean_/singular_values_ vs sklearn (align_rows)"]
fn pca_components_mean_singular_values_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    assert_len(&case, "components_", N_COMPONENTS * N_FEATURES);
    assert_len(&case, "mean_", N_FEATURES);
    assert_len(&case, "singular_values_", N_COMPONENTS);
}

/// `explained_variance_` + `explained_variance_ratio_` vs sklearn, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires explained_variance_/ratio vs sklearn"]
fn pca_explained_variance_ratio_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    assert_len(&case, "explained_variance_", N_COMPONENTS);
    assert_len(&case, "explained_variance_ratio_", N_COMPONENTS);
}

/// `explained_variance_` + `explained_variance_ratio_` vs sklearn, f64.
#[test]
#[ignore = "04-04 removes #[ignore] and wires explained_variance_/ratio vs sklearn"]
fn pca_explained_variance_ratio_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca ev f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    assert_len(&case, "explained_variance_", N_COMPONENTS);
    assert_len(&case, "explained_variance_ratio_", N_COMPONENTS);
}

/// `transform(X)` vs sklearn after `align_rows`, f32.
#[test]
#[ignore = "04-04 removes #[ignore] and wires transform(X) vs sklearn (align_rows)"]
fn pca_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    assert_len(&case, "transform", N_SAMPLES * N_COMPONENTS);
}

/// `transform(X)` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-04 removes #[ignore] and wires transform(X) vs sklearn (align_rows)"]
fn pca_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca transform f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    assert_len(&case, "transform", N_SAMPLES * N_COMPONENTS);
}

/// `inverse_transform(transform(X)) ≈ X`, f32 (PCA-only round-trip, D-01).
#[test]
#[ignore = "04-04 removes #[ignore] and wires inverse_transform round-trip vs sklearn"]
fn pca_inverse_transform_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("pca_f32_seed42.npz")).expect("load pca_f32");
    assert_len(&case, "X", N_SAMPLES * N_FEATURES);
    assert_len(&case, "transform", N_SAMPLES * N_COMPONENTS);
}

/// `inverse_transform(transform(X)) ≈ X`, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-04 removes #[ignore] and wires inverse_transform round-trip vs sklearn"]
fn pca_inverse_transform_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("pca inverse f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("pca_f64_seed42.npz")).expect("load pca_f64");
    assert_len(&case, "X", N_SAMPLES * N_FEATURES);
}
