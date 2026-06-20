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

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, load_npz, OracleCase, Tolerance, F64_TOL};

/// kernel_matrix fixture geometry (gen_oracle.py `KM_ROWS_X` × `KM_COLS`,
/// `KM_ROWS_Y` × `KM_COLS`): K is `rows_x × rows_y`.
const ROWS_X: usize = 5;
const ROWS_Y: usize = 4;
const COLS: usize = 3;

/// Documented f32 band for the PRIM-08 kernel matrix (set FROM the measurement
/// printed by the Wave-1 value test). f64 stays strict `F64_TOL` (1e-5). Carried
/// here so the Wave-1 plan only flips `#[ignore]`.
const KM_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

/// Build an `F` (f32/f64) from an `f64` (mirrors incremental_svd_test::from_f64).
fn from_f64<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_matrix is f32/f64 only"),
    }
}

/// Compute `K(X, Y)` on the device for `kernel`, read it back to host `f64`.
fn compute_km<F>(case: &OracleCase, kernel: Kernel<F>) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let x_f: Vec<F> = case.expect_f64("X").iter().map(|&v| from_f64::<F>(v)).collect();
    let y_f: Vec<F> = case.expect_f64("Y").iter().map(|&v| from_f64::<F>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);

    let k = kernel_matrix::<F>(
        &mut pool,
        &x_dev,
        (ROWS_X, COLS),
        &y_dev,
        (ROWS_Y, COLS),
        kernel,
        None,
    )
    .expect("kernel_matrix computes");

    let host: Vec<F> = k.to_host(&pool);
    // Reinterpret F -> f64 for the comparison (works for both f32 and f64).
    host.iter()
        .map(|&v| match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
            _ => unreachable!(),
        })
        .collect()
}

/// Resolved-default γ = 1/n_features (sklearn `gamma=None` default, D-05) at the
/// test precision; matches `gen_oracle.py`'s `gamma_default = 1.0/KM_COLS`.
fn gamma_default<F: Pod>() -> F {
    from_f64::<F>(1.0 / COLS as f64)
}

/// Drive all four kernels (linear/rbf/poly/sigmoid) at precision `F` and assert
/// each against its committed sklearn `pairwise_kernels` reference within `tol`.
/// The kernel hyperparameters match the fixture generator exactly: γ = 1/cols
/// (default), degree = 3, coef0 = 1 (the sklearn poly/sigmoid defaults).
fn run_all_kernels<F>(case: &OracleCase, tol: &Tolerance, dtype: &str)
where
    F: Float + CubeElement + Pod,
{
    let degree = from_f64::<F>(3.0);
    let coef0 = from_f64::<F>(1.0);
    let gamma = gamma_default::<F>();

    let cases: [(&str, Kernel<F>); 4] = [
        ("K_linear", Kernel::Linear),
        ("K_rbf", Kernel::Rbf { gamma }),
        ("K_poly", Kernel::Poly { gamma, degree, coef0 }),
        ("K_sigmoid", Kernel::Sigmoid { gamma, coef0 }),
    ];

    for (name, kernel) in cases {
        let got = compute_km::<F>(case, kernel);
        let expected = case.expect_f64(name);
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        for (g, &e) in got.iter().zip(expected.iter()) {
            let abs = (g - e).abs();
            max_abs = max_abs.max(abs);
            if e.abs() > 1e-8 {
                max_rel = max_rel.max(abs / e.abs());
            }
        }
        println!(
            "kernel_matrix[{dtype}] {name}: max_abs={max_abs:e} max_rel={max_rel:e} \
             (tol.abs={:e} tol.rel={:e})",
            tol.abs, tol.rel
        );
        assert_slice_close(&got, expected, tol);
    }
}

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
fn kernel_matrix_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_matrix f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = assert_fixture_well_formed("kernel_matrix_f64_seed42.npz");
    run_all_kernels::<f64>(&case, &F64_TOL, "f64");
}

/// PRIM-08 kernel-matrix values vs sklearn at the documented f32 band
/// (`KM_F32_BAND`). Runs on every backend (the f32 gate is rocm; cpu also
/// exercises f32). Wave-1 (08-02) removes `#[ignore]` and wires the real call.
#[test]
fn kernel_matrix_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = assert_fixture_well_formed("kernel_matrix_f32_seed42.npz");
    run_all_kernels::<f32>(&case, &KM_F32_BAND, "f32");
}

/// PoolStats memory gate for `kernel_matrix.rs` (PRIM-08): driving `kernel_matrix`
/// N times at a fixed shape releases the per-call base-op scratch — `live_bytes`
/// conserves after warmup and `peak_bytes` plateaus (the D-10 one-gate-per-prim
/// precedent, mirror `incremental_svd_memory_gate`). Wave-1 (08-02) removes
/// `#[ignore]` and drives the real `kernel_matrix`.
#[test]
fn kernel_matrix_memory_gate() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    const N: usize = 5;
    // Fixed square shape so the distance base AND the in-place map both run at a
    // constant n×n footprint each call. Rbf is chosen so BOTH the distance base op
    // and the rbf_map exercise the pool (the in-place map reuses the base buffer —
    // no parallel allocation, T-08-02-02).
    let rows = 8usize;
    let cols = 4usize;
    let gamma = 0.5f32;

    // Deterministic input (the gate asserts on POOL COUNTERS, not values).
    let make = |seed: usize| -> Vec<f32> {
        (0..rows * cols)
            .map(|i| (((i + seed) % 11) as f32) * 0.1 - 0.5)
            .collect()
    };

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let mut live_after: Vec<u64> = Vec::with_capacity(N);
    let mut peak_after: Vec<u64> = Vec::with_capacity(N);

    for iter in 0..N {
        let x = make(iter);
        let y = make(iter + 3);
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

        let k = kernel_matrix::<f32>(
            &mut pool,
            &x_dev,
            (rows, cols),
            &y_dev,
            (rows, cols),
            Kernel::Rbf { gamma },
            None,
        )
        .expect("kernel_matrix in memory gate");

        // Release this call's transient operands + the produced K so the steady
        // state is the empty free-list (the prim itself releases its internal
        // distance scratch; the in-place map allocates nothing).
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
        k.release_into(&mut pool);

        let stats = pool.stats();
        live_after.push(stats.live_bytes);
        peak_after.push(stats.peak_bytes);
    }

    // After a warmup iteration the live footprint must CONSERVE: the distance
    // base op releases its own XYᵀ / norm scratch and the rbf_map runs IN PLACE
    // over the base buffer (no parallel n×n allocation). A monotone climb is the
    // RED-if-removed signal that a release went missing (build-failing).
    for w in 2..N {
        assert!(
            live_after[w] <= live_after[1],
            "live_bytes must not grow after warmup: iter {w} = {} > iter 1 = {}",
            live_after[w],
            live_after[1]
        );
    }
    // peak_bytes plateaus after warmup (released scratch reused in place).
    for w in 2..N {
        assert_eq!(
            peak_after[w], peak_after[N - 1],
            "peak_bytes must plateau after warmup (iter {w} vs final)"
        );
    }

    println!(
        "kernel_matrix_memory_gate backend={backend}: live={:?} peak={:?} (N={N})",
        live_after, peak_after
    );
}
