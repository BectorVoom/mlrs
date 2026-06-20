//! Plan 08-02 — pairwise kernel-matrix primitive (PRIM-08) tests.
//!
//! **Wave-0 Nyquist scaffold (08-01).** Every function below is `#[ignore]`d and
//! asserts ONLY that its committed `kernel_matrix_*` oracle fixture loads and is
//! shape-well-formed (the `X`/`Y` inputs and the per-kernel reference matrices
//! `K_linear`/`K_rbf`/`K_poly`/`K_sigmoid`). It does NOT reference the
//! `kernel_matrix` compute body beyond the `Kernel<F>` enum + `kernel_matrix`
//! host-fn signature the 08-01 stub already exposes (the stub's compute path is
//! `todo!()`), so this test crate COMPILES today. The Wave-1 plan (08-02) removes
//! `#[ignore]`, wires the real `kernel_matrix` call, and asserts the values vs the
//! sklearn `pairwise_kernels` reference within the 1e-5 abs+rel contract (f64
//! strict `F64_TOL`; f32 a documented per-family band) plus the PoolStats memory
//! gate.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-backend/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase, Tolerance, F64_TOL};

/// kernel_matrix fixture geometry (gen_oracle.py `KM_ROWS_X` × `KM_COLS`,
/// `KM_ROWS_Y` × `KM_COLS`): K is `rows_x × rows_y`.
const ROWS_X: usize = 5;
const ROWS_Y: usize = 4;
const COLS: usize = 3;

/// Documented f32 band for the PRIM-08 kernel matrix (set FROM the measurement
/// printed by the Wave-1 value test). f64 stays strict `F64_TOL` (1e-5). Carried
/// here so the Wave-1 plan only flips `#[ignore]`.
#[allow(dead_code)]
const KM_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Load a kernel_matrix oracle blob and assert the inputs + all four per-kernel
/// reference matrices are shape-well-formed (the Wave-0 contract: fixture-load +
/// shape only, no compute).
fn assert_fixture_well_formed(name: &str) -> OracleCase {
    let case = load_npz(fixture(name)).expect("kernel_matrix fixture loads");
    assert_eq!(
        case.expect_f64("X").len(),
        ROWS_X * COLS,
        "X is rows_x × cols"
    );
    assert_eq!(
        case.expect_f64("Y").len(),
        ROWS_Y * COLS,
        "Y is rows_y × cols"
    );
    for k in ["K_linear", "K_rbf", "K_rbf_gamma", "K_poly", "K_sigmoid"] {
        assert_eq!(
            case.expect_f64(k).len(),
            ROWS_X * ROWS_Y,
            "{k} is rows_x × rows_y"
        );
    }
    case
}

/// PRIM-08 kernel-matrix values vs sklearn `pairwise_kernels` (linear/rbf/poly/
/// sigmoid), f64 strict `F64_TOL`. Gated by `skip_f64_with_log` (cpu runs; rocm
/// skips). Wave-1 (08-02) removes `#[ignore]` and wires the real `kernel_matrix`.
#[test]
#[ignore = "Wave-0 scaffold: kernel_matrix compute path lands in plan 08-02"]
fn kernel_matrix_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_matrix f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let _ = (assert_fixture_well_formed("kernel_matrix_f64_seed42.npz"), &F64_TOL);
}

/// PRIM-08 kernel-matrix values vs sklearn at the documented f32 band
/// (`KM_F32_BAND`). Runs on every backend (the f32 gate is rocm; cpu also
/// exercises f32). Wave-1 (08-02) removes `#[ignore]` and wires the real call.
#[test]
#[ignore = "Wave-0 scaffold: kernel_matrix compute path lands in plan 08-02"]
fn kernel_matrix_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let _ = assert_fixture_well_formed("kernel_matrix_f32_seed42.npz");
}

/// PoolStats memory gate for `kernel_matrix.rs` (PRIM-08): driving `kernel_matrix`
/// N times at a fixed shape releases the per-call base-op scratch — `live_bytes`
/// conserves after warmup and `peak_bytes` plateaus (the D-10 one-gate-per-prim
/// precedent, mirror `incremental_svd_memory_gate`). Wave-1 (08-02) removes
/// `#[ignore]` and drives the real `kernel_matrix`.
#[test]
#[ignore = "Wave-0 scaffold: kernel_matrix memory gate lands in plan 08-02"]
fn kernel_matrix_memory_gate() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    // Wave-1: drive kernel_matrix N times at a fixed shape; assert
    // live_after[w] <= live_after[1] and peak_after plateaus.
    println!("kernel_matrix_memory_gate backend={backend}: scaffold (no-op until 08-02)");
}
