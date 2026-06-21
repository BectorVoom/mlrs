//! Plan 09-01 — SpectralEmbedding (SPECTRAL-01) sklearn oracle scaffolds.
//!
//! Wave-0 Nyquist `#[ignore]` scaffolds: each test loads its committed fixture
//! and asserts fixture-load + SHAPE only (the estimator `fit` / `embedding_`
//! bodies are `todo!()` until the Wave-2 plan 09-03, which un-ignores these and
//! fills the value-match-after-sign-align / subspace / reject-oversize
//! assertions). They compile + collect today against the Wave-0 stubs.
//!
//! Case map (9-SE-01..04, un-ignored by 09-03):
//!   - `spectral_embedding` — rbf affinity value-match after sign alignment.
//!   - `knn_affinity` — `nearest_neighbors` default affinity (D-01/D-03).
//!   - `subspace` — degenerate-spectrum subspace test (principal angles, D-09).
//!   - `reject_oversize` — `n_samples > 64` → `AlgoError::NSamplesExceedsMaxDim`
//!     BEFORE any device work (D-06).
//!
//! f64 carries the `skip_f64_with_log` gate verbatim; f32 runs on rocm at the
//! documented `SE_F32_BAND`. Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_algos::error::AlgoError;
use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase, Tolerance};

/// SpectralEmbedding fixture geometry (gen_oracle.py `SE_N_SAMPLES` ×
/// `SE_N_FEATURES`, `SE_N_COMPONENTS` embedding columns).
const N_SAMPLES: usize = 12;
const N_COMPONENTS: usize = 2;

/// Documented f32 band for the SPECTRAL-01 embedding (the v1 per-family
/// documented-band precedent; the strict 1e-5 absolute arm is never loosened).
/// f64 stays strict `F64_TOL` (1e-5). The observed max f32 error is recorded in
/// the SUMMARY when 09-03 un-ignores.
const SE_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the fixture exposes `X` + the reference `embedding_` (n × n_components)
/// (shape-only Wave-0 scaffold; the value compare lands when 09-03 un-ignores).
fn assert_shapes(case: &OracleCase, n: usize, k: usize) {
    assert!(!case.expect_f64("X").is_empty(), "X must be present");
    assert_eq!(
        case.expect_f64("embedding").len(),
        n * k,
        "embedding must be n × n_components"
    );
}

/// 9-SE-01: rbf-affinity embedding value-match after sign alignment, f64 strict.
/// Gated by `skip_f64_with_log`. (Wave-0 scaffold: fixture-load + shape only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + value-asserted by plan 09-03"]
fn spectral_embedding() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_f64_seed42.npz"))
        .expect("load spectral_embedding_f64");
    assert_shapes(&case, N_SAMPLES, N_COMPONENTS);
    let _ = &SE_F32_BAND; // band kept load-bearing for the 09-03 f32 path.
}

/// 9-SE-02: `nearest_neighbors` default-affinity embedding (D-01/D-03).
/// (Wave-0 scaffold: fixture-load + shape only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + value-asserted by plan 09-03"]
fn knn_affinity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding knn f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_f64_seed42.npz"))
        .expect("load spectral_embedding_f64");
    assert_shapes(&case, N_SAMPLES, N_COMPONENTS);
}

/// 9-SE-03: degenerate-spectrum subspace test (principal angles, D-09).
/// (Wave-0 scaffold: fixture-load + shape only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + subspace-asserted by plan 09-03"]
fn subspace() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_embedding subspace f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_embedding_degenerate_f64_seed42.npz"))
        .expect("load spectral_embedding_degenerate_f64");
    assert_shapes(&case, N_SAMPLES, N_COMPONENTS);
}

/// 9-SE-04: `n_samples > 64` is rejected with `AlgoError::NSamplesExceedsMaxDim`
/// BEFORE any device work (D-06). (Wave-0 scaffold: the typed variant exists +
/// is constructible with the spectral cap message; the live `fit`-rejection
/// assertion lands when 09-03 un-ignores.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + fit-rejection-asserted by plan 09-03"]
fn reject_oversize() {
    // The typed guard the Wave-2 fit will raise BEFORE any affinity/Laplacian/eig
    // launch exists and names the MAX_DIM=64 cap (D-06). Asserted structurally
    // here; the live `fit(n=65) -> Err(NSamplesExceedsMaxDim)` path lands in 09-03.
    let err = AlgoError::NSamplesExceedsMaxDim {
        estimator: "spectral_embedding",
        n_samples: 65,
        max: 64,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("65") && msg.contains("64") && msg.contains("MAX_DIM"),
        "NSamplesExceedsMaxDim message must name the offending n_samples + the cap: {msg}"
    );
}
