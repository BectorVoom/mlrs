//! Plan 07-03 — incremental (batched) SVD merge primitive (PRIM-07) tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness — it makes
//! NO reference to the not-yet-existent `mlrs_backend::prims::incremental_svd`
//! symbol (the `incremental_svd.rs` body is an empty stub until plan 07-03). This
//! is the 03-02 / 05-01 Wave-0 pattern: the test crate must COMPILE today; plan
//! 07-03 removes the `#[ignore]`, wires the real stacked-matrix `merge` calls,
//! and turns each stub into the live oracle/invariant assertion.
//!
//! PRIM-07 gate (plan 07-03 wires): a 2+-batch incremental merge whose running
//! `(singular_values_, components_, mean_)` is compared to a HOST reference
//! computed from a single full-batch SVD of the same data (ddof=1; `align_rows`
//! sign-canonicalization applied after every batch — == sklearn
//! `svd_flip(u_based_decision=False)`), plus a PoolStats memory gate. The
//! `incremental_pca_*` oracle blobs (committed by plan 07-01's gen_oracle.py) are
//! the value source — the SVD merge is sized so the stacked matrix clears the
//! Phase-3 caps (`n_components + batch_size + 1 ≤ MAX_ROWS`, `n_features ≤
//! MAX_COLS`). Fixtures are kept TINY (the SVD path is the slow one — cpu suite
//! ~6 min).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim from
//! `gemm_test.rs` (cpu runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2
//! tests live in `crates/mlrs-backend/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::load_npz;

/// Resolve a workspace-root-relative fixture path (matches `svd_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// 2+-batch incremental merge vs a single-pass host SVD reference (PRIM-07):
/// `components_` (after `align_rows`) + `singular_values_` match the full-batch
/// SVD within 1e-5 (f64) / a documented f32 band. f64 gated by `skip_f64_with_log`.
///
/// WAVE-0 STUB: loads the `incremental_pca_nowhiten_f64` blob and asserts its
/// `X` + `components_` shapes are well-formed (the stacked merge clears the caps).
/// Plan 07-03 removes `#[ignore]` and wires `incremental_svd::merge` over 2+
/// batches vs the host reference.
#[test]
#[ignore = "wave-0 scaffold: prims::incremental_svd lands in plan 07-03"]
fn incremental_svd_two_batch_merge() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_svd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");

    // Shape well-formedness only (no prim call yet): X is (n × p), components_ is
    // (n_components × p), and the stacked SVD matrix must clear the Phase-3 caps.
    let x_shape = case.shape("X").expect("X shape").to_vec();
    let c_shape = case.shape("components_").expect("components_ shape").to_vec();
    assert_eq!(x_shape.len(), 2, "X is a 2-D matrix");
    assert_eq!(c_shape.len(), 2, "components_ is a 2-D matrix");
    let n_features = x_shape[1] as usize;
    let n_components = c_shape[0] as usize;
    assert_eq!(c_shape[1] as usize, n_features, "components_ cols == n_features");

    let batch_size = case.expect_f64("batch_size")[0] as usize;
    assert!(
        n_components + batch_size + 1 <= 256,
        "stacked SVD rows (nc+bs+1={}) must clear MAX_ROWS=256",
        n_components + batch_size + 1
    );
    assert!(n_features <= 64, "n_features ({n_features}) must clear MAX_COLS=64");
}

/// PoolStats memory gate for `incremental_svd.rs` (PRIM-07): the per-batch SVD
/// scratch is released back to the pool — allocations bounded across the batch
/// stream (the D-10 one-gate-per-prim precedent; `live_bytes`/`peak_bytes`/
/// `reuses` PoolStats idiom).
///
/// WAVE-0 STUB. Plan 07-03 wires the multi-batch `incremental_svd::merge` stream
/// and asserts `pool.stats()` allocation/reuse bounds.
#[test]
#[ignore = "wave-0 scaffold: prims::incremental_svd + memory gate land in plan 07-03"]
fn incremental_svd_memory_gate() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    assert_eq!(
        case.shape("X").expect("X shape").len(),
        2,
        "X is a 2-D matrix"
    );
}
