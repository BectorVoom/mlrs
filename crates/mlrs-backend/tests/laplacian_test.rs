//! Plan 09-01 — laplacian (PRIM-09) sklearn/host-reference oracle scaffolds.
//!
//! These are the Wave-0 Nyquist `#[ignore]` scaffolds: each test loads its
//! committed fixture and asserts fixture-load + SHAPE only (no laplacian compute
//! symbols — the prim's compute path is `todo!()` until the Wave-1 plan 09-02,
//! which un-ignores these and fills the value / zero-degree / memory-gate
//! assertions). They compile + collect today against the Wave-0 stubs.
//!
//! Case map (RESEARCH Test Map, un-ignored by 09-02):
//!   - `laplacian_value` — `L = I − D^-1/2 A D^-1/2` vs a host reference, f32+f64.
//!   - `zero_degree` — no NaN / no infinite value on an isolated-node fixture.
//!   - `memory_gate` — `BufferPool` / `PoolStats` reuse-bounded gate (mirrors
//!     `memory_gate_test.rs`).
//!
//! f64 carries the `skip_f64_with_log` capability gate verbatim (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-backend/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// laplacian fixture geometry (gen_oracle.py `LAP_N` × `LAP_N`).
const N: usize = 8;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the fixture exposes a well-formed `n×n` affinity + reference Laplacian
/// (shape-only Wave-0 scaffold; the value compare lands when 09-02 un-ignores).
fn assert_shapes(case: &OracleCase, n: usize) {
    assert_eq!(case.expect_f64("A").len(), n * n, "A must be n×n");
    assert_eq!(case.expect_f64("L").len(), n * n, "L must be n×n");
    assert_eq!(case.expect_f64("dd").len(), n, "dd must be length n");
}

/// PRIM-09 normalized Laplacian vs host reference, f64 strict. Gated by
/// `skip_f64_with_log`. (Wave-0 scaffold: fixture-load + shape only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + value-asserted by plan 09-02"]
fn laplacian_value() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("laplacian f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("laplacian_f64_seed42.npz")).expect("load laplacian_f64");
    assert_shapes(&case, N);
}

/// PRIM-09 no-NaN / no-infinite-value on a zero-degree (isolated-node) graph —
/// the typed-zero guard success criterion. (Wave-0 scaffold: shape only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + value-asserted by plan 09-02"]
fn zero_degree() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("laplacian zero_degree f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case =
        load_npz(fixture("laplacian_isolated_f64_seed42.npz")).expect("load laplacian_isolated_f64");
    assert_shapes(&case, N);
    // Every reference Laplacian entry is finite (the zero-degree guard never
    // emits a NaN or an infinite value); asserted on the committed reference here,
    // and on the device output once 09-02 un-ignores.
    for &v in case.expect_f64("L") {
        assert!(v.is_finite(), "reference L must be finite on isolated nodes");
    }
}

/// PRIM-09 `BufferPool` / `PoolStats` reuse-bounded memory gate (mirrors
/// `memory_gate_test.rs`). (Wave-0 scaffold: pool counters API only.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + counter-asserted by plan 09-02"]
fn memory_gate() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    let _ = backend;
    let client = runtime::active_client();
    let pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    // The pool counters API is live at construction (allocations / reuses /
    // live_bytes / peak_bytes); the hard reuse-bounded assertions over repeated
    // `laplacian` calls land when 09-02 un-ignores this gate.
    let stats = pool.stats();
    assert_eq!(stats.live_bytes, 0, "fresh pool has zero live bytes");
}
