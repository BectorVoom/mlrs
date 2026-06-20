//! Plan 07-05 — IncrementalPCA (DECOMP-03) sklearn oracle tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness — it makes
//! NO reference to the not-yet-existent
//! `mlrs_algos::decomposition::incremental_pca` estimator nor the `PartialFit`
//! call (the module is created by plan 07-05). This is the 04-01 / 05-01 Wave-0
//! pattern: the test crate must COMPILE today; plan 07-05 removes the `#[ignore]`,
//! wires the real `IncrementalPCA::partial_fit`/`fit`, and turns each stub into
//! the 1e-5 oracle compare (post `align_rows`).
//!
//! The DECOMP-03 gate (plan 07-05 wires): all attributes (`components_`,
//! `explained_variance_`, `explained_variance_ratio_`, `singular_values_`,
//! `mean_`, `var_`, `n_samples_seen_`) + `transform`/`inverse_transform` vs
//! sklearn IncrementalPCA, fitted BOTH via `partial_fit` over batches AND via the
//! one-shot `fit()` (D-02 — `fit` loops `partial_fit` over `gen_batches`),
//! `whiten` on/off, compared AFTER `align_rows` (== sklearn `svd_flip(
//! u_based_decision=False)`).
//!
//! Two fixtures per dtype: `incremental_pca_nowhiten_*` and
//! `incremental_pca_whiten_*` (30×6, n_components=3, batch_size=10 — the stacked
//! per-batch SVD matrix `nc+bs+1=14 ≤ 256`, `n_features=6 ≤ 64`).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band
//! (`IPCA_F32_BAND` — Claude's-discretion, pinned in plan 07-05). Per AGENTS.md
//! §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, Tolerance, F32_TOL, F64_TOL};

/// IncrementalPCA fixture geometry (gen_oracle.py `IPCA_SHAPE` = 30×6).
const IPCA_N: usize = 30;
const IPCA_P: usize = 6;
const IPCA_N_COMPONENTS: usize = 3;
const IPCA_BATCH_SIZE: usize = 10;

/// f32-on-rocm per-family tolerance band for IncrementalPCA, pinned from the
/// standalone-estimator measurement in plan 07-05 (Claude's-discretion, D-08
/// growth point — the streaming SVD merge accumulates f32 round-off). f64 stays
/// strict `F64_TOL` (1e-5); this is the f32 placeholder plan 07-05 replaces.
#[allow(dead_code)]
const IPCA_F32_BAND: Tolerance = F32_TOL;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the IncrementalPCA fixture's attribute shapes are well-formed (no
/// estimator call yet): `X` is (n × p), `components_` is (nc × p), the streaming
/// attrs (`mean_`/`var_`) are length p, `n_samples_seen_` == n, and the stacked
/// per-batch SVD matrix clears the Phase-3 caps.
fn assert_ipca_shapes(case: &mlrs_core::OracleCase) {
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![IPCA_N as u64, IPCA_P as u64]);
    assert_eq!(
        case.shape("components_").expect("components_").to_vec(),
        vec![IPCA_N_COMPONENTS as u64, IPCA_P as u64]
    );
    assert_eq!(case.expect_f64("mean_").len(), IPCA_P);
    assert_eq!(case.expect_f64("var_").len(), IPCA_P);
    assert_eq!(case.expect_f64("n_samples_seen_")[0] as usize, IPCA_N);
    assert_eq!(case.expect_f64("singular_values_").len(), IPCA_N_COMPONENTS);
    // transform is (n × nc); inverse_transform is (n × p).
    assert_eq!(case.expect_f64("transform").len(), IPCA_N * IPCA_N_COMPONENTS);
    assert_eq!(case.expect_f64("inverse_transform").len(), IPCA_N * IPCA_P);
    assert!(
        IPCA_N_COMPONENTS + IPCA_BATCH_SIZE + 1 <= 256,
        "stacked SVD rows must clear MAX_ROWS=256"
    );
    assert!(IPCA_P <= 64, "n_features must clear MAX_COLS=64");
}

/// All attrs + transform/inverse_transform via `partial_fit` AND `fit()`,
/// whiten=False, f32 (cpu + rocm), compared after `align_rows`.
///
/// WAVE-0 STUB. Plan 07-05 removes `#[ignore]` and wires both the `partial_fit`
/// stream and the one-shot `fit()` → 1e-5/`IPCA_F32_BAND` compare.
#[test]
#[ignore = "wave-0 scaffold: decomposition::incremental_pca lands in plan 07-05"]
fn incremental_pca_nowhiten_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    assert_eq!(case.expect_f64("whiten")[0] as usize, 0, "nowhiten fixture");
    assert_ipca_shapes(&case);
}

/// All attrs + transform/inverse_transform via `partial_fit` AND `fit()`,
/// whiten=False, f64 (cpu runs; rocm skips-with-log), after `align_rows`.
///
/// WAVE-0 STUB. Plan 07-05 wires the f64 partial_fit + fit 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: decomposition::incremental_pca lands in plan 07-05"]
fn incremental_pca_nowhiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca nowhiten f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    assert_eq!(case.expect_f64("whiten")[0] as usize, 0, "nowhiten fixture");
    assert_ipca_shapes(&case);
    let _ = &F64_TOL; // 1e-5 contract used by plan 07-05's compare.
}

/// All attrs + transform/inverse_transform with whiten=True, f32 (cpu + rocm) —
/// whiten scales components by `1/sqrt(explained_variance_)` (D-06).
///
/// WAVE-0 STUB. Plan 07-05 wires the whiten=True f32 compare.
#[test]
#[ignore = "wave-0 scaffold: decomposition::incremental_pca lands in plan 07-05"]
fn incremental_pca_whiten_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_whiten_f32_seed42.npz"))
        .expect("load incremental_pca_whiten_f32");
    assert_eq!(case.expect_f64("whiten")[0] as usize, 1, "whiten fixture");
    assert_ipca_shapes(&case);
}

/// All attrs + transform/inverse_transform with whiten=True, f64 (cpu runs; rocm
/// skips-with-log) — whiten/un-whiten round-trip in inverse_transform.
///
/// WAVE-0 STUB. Plan 07-05 wires the whiten=True f64 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: decomposition::incremental_pca lands in plan 07-05"]
fn incremental_pca_whiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca whiten f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_whiten_f64_seed42.npz"))
        .expect("load incremental_pca_whiten_f64");
    assert_eq!(case.expect_f64("whiten")[0] as usize, 1, "whiten fixture");
    assert_ipca_shapes(&case);
}
