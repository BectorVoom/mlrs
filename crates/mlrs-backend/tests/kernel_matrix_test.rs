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
use mlrs_backend::prims::launch_dims_1d_folded;
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


// ===========================================================================
// Launch geometry for the in-place kernel maps (`rbf_map` / `poly_map` /
// `sigmoid_map`).
//
// The map element count is PAIRWISE (`rows_x · rows_y`), which is what makes
// the grid cap reachable at ordinary training sizes: at the GPU-idiomatic 256
// width an `n × n` Gram needs more than `MAX_GRID_DIM` cubes from n = 4097, and
// at a 16-unit cpu width from n = 1025. The map launch used to be a private
// single-axis `ceil(n / 256)` copy, which has no way to express a grid past
// that point; it now goes through the shared FOLDED helper at the backend's own
// gather width.
// ===========================================================================

/// Local copy of the per-dimension cube cap (`mlrs_kernels::colmean::MAX_GRID_DIM`
/// is not on the backend's public test surface — the `center_test` local-cap
/// precedent, kept in sync by the assertions it drives).
const MAX_GRID_DIM_TEST: u32 = 65_535;

/// The folded map geometry must (a) keep every grid dimension inside the cap and
/// (b) still provision at least one unit per element, at every width a backend
/// can hand it and across the `n = 4096 / 4097` boundary where the old
/// single-axis form ran out of grid.
///
/// RED if the map launch is moved back to an unfolded `ceil(n / block)`: the
/// pairwise rungs below produce cube counts of 65 536 and 4 194 304, both past
/// the cap, and the first assertion fails.
#[test]
fn kernel_map_launch_geometry_folds_within_grid_cap() {
    // Widths a backend actually supplies: `gather_launch_width()` is
    // `cpu_launch_units()` on cpu (core count, 1..=64 in practice) and 256
    // elsewhere.
    for block in [1u32, 4, 8, 16, 32, 64, 256] {
        for n in [
            1usize,
            block as usize + 1,        // one past a single full cube
            1024 * 1024,               // n=1024 square Gram
            1025 * 1025,               // one past the 16-width cap
            4096 * 4096,               // exactly at the 256-width cap
            4097 * 4097,               // one past it — the old form's ceiling
            1 << 30,
        ] {
            let (count, dim) = launch_dims_1d_folded(n, block);
            let (cx, cy, cz) = match count {
                CubeCount::Static(x, y, z) => (x, y, z),
                other => panic!("expected a static cube count, got {other:?}"),
            };
            assert!(
                cx <= MAX_GRID_DIM_TEST && cy <= MAX_GRID_DIM_TEST && cz <= MAX_GRID_DIM_TEST,
                "grid dimension exceeds the {MAX_GRID_DIM_TEST} cap for n={n} block={block}: \
                 ({cx}, {cy}, {cz})"
            );
            assert_eq!(dim.x, block, "cube width must be the requested one");
            assert_eq!((dim.y, dim.z), (1, 1), "the map launch is 1-D in the cube");

            // Coverage: the flattened `ABSOLUTE_POS` space must reach every
            // element, or the trailing ones are silently never mapped.
            let units = (cx as u64) * (cy as u64) * (cz as u64) * (dim.x as u64);
            assert!(
                units >= n as u64,
                "under-provisioned grid for n={n} block={block}: {units} units < {n} elements"
            );
        }
    }
}

/// End-to-end value check on an `n × n` rbf Gram whose element count needs the
/// grid fold: `n = 4097` gives 16 785 409 elements, i.e. 65 568 cubes at the 256
/// width — 33 past the single-dimension cap the old private launch helper had no
/// second axis to spill into.
///
/// `cols = 1` keeps the `O(n²·d)` distance base as cheap as the fold test allows
/// while leaving the map itself at full size, and lets the expected value be
/// written in closed form. The LAST element is the one that matters: an
/// under-provisioned grid drops the tail, so a correct `K[n-1][n-1]` is what
/// separates a real fold from a launch that quietly stopped early.
#[test]
#[ignore = "heavy ~16.8M-element Gram — run explicitly to verify the map grid fold"]
fn kernel_matrix_rbf_grid_fold_large_n_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    // One past the 256-width single-dimension cap (65 536 cubes > 65 535).
    let n = 4097usize;
    let cols = 1usize;
    let gamma = 0.25f32;

    // x[i] = i / n, so ‖xᵢ − xⱼ‖² = ((i − j)/n)² and K[i][j] = exp(−γ·that).
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    let k = kernel_matrix::<f32>(
        &mut pool,
        &x_dev,
        (n, cols),
        &x_dev,
        (n, cols),
        Kernel::Rbf { gamma },
        None,
    )
    .expect("kernel_matrix accepts a valid large square shape");
    let got = k.to_host(&pool);

    let expect = |i: usize, j: usize| -> f64 {
        let d = (i as f64 - j as f64) / n as f64;
        (-(gamma as f64) * d * d).exp()
    };
    // First, a mid-matrix element, and — the one that actually pins the fold —
    // the very last, which lives in the cube the unfolded grid could not address.
    for (i, j) in [(0usize, 0usize), (n / 2, n / 3), (n - 1, n - 1), (n - 1, 0)] {
        let g = got[i * n + j] as f64;
        let e = expect(i, j);
        assert!(
            (g - e).abs() <= 1e-5,
            "K[{i}][{j}] wrong (tail cubes dropped?): got={g:e} expected={e:e} backend={backend}"
        );
    }
    println!(
        "kernel_matrix_rbf_grid_fold_large_n_f32 backend={backend}: \
         n={n} elems={} (65568 cubes at width 256) OK",
        n * n
    );
}
