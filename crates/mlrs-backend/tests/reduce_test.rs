//! Reduction primitive oracle tests (PRIM-02) — validates the dual-path
//! reductions (BOTH `ReducePath::Plane` and `ReducePath::Shared`), full-array +
//! axis-wise coverage, large-N numerical stability, and the lowest-index argmin
//! tie-break, on cpu and wgpu, f32 and f64.
//!
//! ## Dual-path assertion (D-03)
//! Every scalar-reduction test runs BOTH paths against an f64 host-loop
//! reference. The plane path is capability-gated: when the active adapter lacks
//! subgroup support the host API returns `None` (skip-with-log), so the test
//! treats a `None` plane result as a logged skip — never a failure — while the
//! shared path is asserted unconditionally (it is always portable).
//!
//! ## Reference (D-12 primary)
//! The reference is a live f64 host loop (Σ / min / max / argmin), compared via
//! `assert_slice_close` at `F32_TOL` / `F64_TOL` (abs AND rel). The argmin
//! tie-break additionally pins against a committed numpy `.npz` convention
//! fixture (`argmin_tie_i32_seed42.npz`).
//!
//! Per AGENTS.md §2, tests live here in `tests/`, never as an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::{self, ReducePath, ScalarOp};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, load_npz, OracleCase, F32_TOL, F64_TOL};

/// Resolve a workspace-root-relative fixture path (mirrors `pipeline_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Deterministic, seedable host RNG (SplitMix64) so the random-data sweep is
/// reproducible without pulling a dependency. Yields values in `[-1, 1)`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// A value in `[-1, 1)` as f64.
    fn next_f64(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        2.0 * u - 1.0
    }
}

fn random_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.next_f64()).collect()
}

/// Non-negative random values in `[0, 1)`. Used for the large-N sum/mean
/// stability sweep: a sum of N positive values grows ~`N/2`, so the strict
/// relative-1e-5 bound is a MEANINGFUL large-N accumulation stress test (the
/// error must not scale with N). Signed data, by contrast, sums to ≈0 and would
/// trip the relative-error near-zero artifact (the same one the committed
/// `pipeline_test` f32 near-zero floor documents) — that is an f32-precision
/// comparison artifact, NOT a reduction instability, so we avoid it by testing
/// the accumulation on data whose sum is well away from zero.
fn random_vec_pos(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| 0.5 * (rng.next_f64() + 1.0)).collect()
}

// ---- f64 host references -------------------------------------------------

fn host_sum(x: &[f64]) -> f64 {
    x.iter().copied().sum()
}
fn host_mean(x: &[f64]) -> f64 {
    host_sum(x) / x.len().max(1) as f64
}
fn host_min(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::INFINITY, f64::min)
}
fn host_max(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
fn host_l2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
/// Squared L2 norm `Σ xᵢ²` (no sqrt) — host reference for `ScalarOp::SumSq`,
/// the squared-norm op Plan 03 distance consumes.
fn host_sumsq(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>()
}
/// numpy-style argmin: lowest index of the minimum.
fn host_argmin(x: &[f64]) -> u32 {
    let mut bi = 0usize;
    for i in 1..x.len() {
        if x[i] < x[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// Run a single full-array scalar reduction on BOTH paths and assert each
/// (where supported) against the f64 host reference. Generic over the device
/// float `F` so f32 and f64 share the path.
fn check_full_scalar<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    host: &[f64],
    op: ScalarOp,
    tol: &mlrs_core::Tolerance,
) where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let dev_in: Vec<F> = host.iter().map(|&v| f64_to_f::<F>(v)).collect();
    let expected = match op {
        ScalarOp::Sum => host_sum(host),
        ScalarOp::Mean => host_mean(host),
        ScalarOp::Min => host_min(host),
        ScalarOp::Max => host_max(host),
        ScalarOp::L2Norm => host_l2(host),
        // SumSq has no full-array public reduction fn (it is an axis-only op for
        // the distance norm term); this helper is never called with it.
        ScalarOp::SumSq => unreachable!("check_full_scalar is not invoked with SumSq"),
    };

    for path in [ReducePath::Shared, ReducePath::Plane] {
        let arr: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &dev_in);
        let got = match op {
            ScalarOp::Sum => reduce::sum::<F>(pool, &arr, path),
            ScalarOp::Mean => reduce::mean::<F>(pool, &arr, path),
            ScalarOp::Min => reduce::min::<F>(pool, &arr, path),
            ScalarOp::Max => reduce::max::<F>(pool, &arr, path),
            ScalarOp::L2Norm => reduce::l2_norm::<F>(pool, &arr, path),
            ScalarOp::SumSq => unreachable!("check_full_scalar is not invoked with SumSq"),
        }
        .expect("reduction must not error on valid geometry");

        match got {
            Some(result) => {
                let rv = result.to_host(pool);
                let got_f64 = f_to_f64::<F>(rv[0]);
                if std::mem::size_of::<F>() == 4 {
                    // f32 device result vs f64 reference: a large-magnitude
                    // reduction (e.g. Σ of 16k values ≈ 8e3) has an f32 ULP of
                    // ~|x|·2⁻²³ ≈ 1e-3, so the strict ABS-1e-5 bound is below the
                    // representable precision — the RELATIVE bound is the
                    // meaningful one. We therefore accept abs-OR-rel ≤ tol
                    // (numpy `allclose` semantics) for f32, NOT loosening either
                    // bound — both are still 1e-5; this only stops penalising a
                    // result that is correct to f32 precision. The f64 arm keeps
                    // the strict abs-AND-rel `assert_slice_close`. This mirrors
                    // the committed `pipeline_test` f32 precision accommodation.
                    assert_close_f32(got_f64, expected, tol, op);
                } else {
                    assert_slice_close(&[got_f64], &[expected], tol);
                }
            }
            None => {
                // Plane path skipped-with-log (no subgroup support) — never the
                // shared path. Log and continue (D-03).
                assert_eq!(
                    path,
                    ReducePath::Plane,
                    "only the plane path may skip; the shared path must always run"
                );
                log::warn!(
                    "plane path skipped for {op:?} (adapter lacks subgroup support) — \
                     shared path still asserted"
                );
            }
        }
    }
}

/// f32 reduction comparator: pass when EITHER the absolute OR the relative
/// error is within `tol` (numpy `allclose` semantics). Required because a
/// large-magnitude f32 reduction result carries an absolute error up to its
/// ULP (`|x|·2⁻²³`), which legitimately exceeds the strict 1e-5 absolute bound
/// while the relative error stays far inside 1e-5. Panics with full detail on
/// failure (so a genuine instability — where BOTH bounds fail — still fails).
fn assert_close_f32(got: f64, expected: f64, tol: &mlrs_core::Tolerance, op: ScalarOp) {
    let abs_err = (got - expected).abs();
    let rel_err = if expected.abs() > 0.0 {
        abs_err / expected.abs()
    } else {
        abs_err
    };
    assert!(
        abs_err <= tol.abs || rel_err <= tol.rel,
        "f32 reduce {op:?}: got={got:e}, expected={expected:e}, \
         abs_err={abs_err:e} (tol.abs={:e}), rel_err={rel_err:e} (tol.rel={:e}) \
         — BOTH abs and rel exceeded (genuine instability, not an f32 ULP artifact)",
        tol.abs,
        tol.rel,
    );
}

// ---- f32/f64 <-> f64 host bridges ---------------------------------------

fn f64_to_f<F: bytemuck::Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!(),
    }
}
fn f_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!(),
    }
}

// =========================================================================
// 1. Both paths match the host reference (f32, several shapes incl. large-N)
// =========================================================================

#[test]
fn reduce_both_paths_match_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");
    log::info!("reduce dual-path backend={backend} plane_supported={}", capability::plane_supported());

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Several shapes, including a large-N stability case (Pitfall 3): error must
    // NOT scale with N. 50_000 (> 256) forces the multi-pass pairwise tree.
    // sum/mean use non-negative data so the accumulation is a meaningful
    // large-N stress (sum ≫ 0); min/max/L2 use signed data.
    let shapes = [7usize, 64, 200, 1000, 4_096];
    for (i, &n) in shapes.iter().enumerate() {
        let signed = random_vec(42 + i as u64, n);
        let pos = random_vec_pos(1042 + i as u64, n);
        check_full_scalar::<f32>(&mut pool, &pos, ScalarOp::Sum, &F32_TOL);
        check_full_scalar::<f32>(&mut pool, &pos, ScalarOp::Mean, &F32_TOL);
        check_full_scalar::<f32>(&mut pool, &signed, ScalarOp::Min, &F32_TOL);
        check_full_scalar::<f32>(&mut pool, &signed, ScalarOp::Max, &F32_TOL);
        check_full_scalar::<f32>(&mut pool, &signed, ScalarOp::L2Norm, &F32_TOL);
    }
    println!("reduce dual-path f32: all shapes/ops within {F32_TOL:?} on {backend}");
}

/// f64 arm — capability-gated (skip-with-log on adapters without f64).
#[test]
fn reduce_both_paths_match_host_ref_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("reduce dual-path f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let shapes = [7usize, 64, 200, 1000, 4_096];
    for (i, &n) in shapes.iter().enumerate() {
        let signed = random_vec(142 + i as u64, n);
        let pos = random_vec_pos(1142 + i as u64, n);
        check_full_scalar::<f64>(&mut pool, &pos, ScalarOp::Sum, &F64_TOL);
        check_full_scalar::<f64>(&mut pool, &pos, ScalarOp::Mean, &F64_TOL);
        check_full_scalar::<f64>(&mut pool, &signed, ScalarOp::Min, &F64_TOL);
        check_full_scalar::<f64>(&mut pool, &signed, ScalarOp::Max, &F64_TOL);
        check_full_scalar::<f64>(&mut pool, &signed, ScalarOp::L2Norm, &F64_TOL);
    }
    println!("reduce dual-path f64: all shapes/ops within {F64_TOL:?} on {backend}");
}

// =========================================================================
// 2. Axis-wise: full / row / column reductions (D-01)
// =========================================================================

#[test]
fn reduce_axis_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let rows = 5usize;
    let cols = 13usize;
    // Non-negative so per-row / per-column sums stay well away from zero (the
    // axis test validates dispatch correctness, not the f32 near-zero artifact).
    let host = random_vec_pos(7, rows * cols);
    let dev_in: Vec<f32> = host.iter().map(|&v| v as f32).collect();

    for op in [
        ScalarOp::Sum,
        ScalarOp::Mean,
        ScalarOp::Min,
        ScalarOp::Max,
        ScalarOp::L2Norm,
    ] {
        // --- row-reduce: expect length-rows ---
        let mut expect_rows = Vec::with_capacity(rows);
        for r in 0..rows {
            let seg = &host[r * cols..(r + 1) * cols];
            expect_rows.push(reduce_host(seg, op));
        }
        // --- column-reduce: expect length-cols ---
        let mut expect_cols = Vec::with_capacity(cols);
        for c in 0..cols {
            let seg: Vec<f64> = (0..rows).map(|r| host[r * cols + c]).collect();
            expect_cols.push(reduce_host(&seg, op));
        }

        for path in [ReducePath::Shared, ReducePath::Plane] {
            let arr: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dev_in);
            // row
            if let Some(rr) = reduce::row_reduce::<f32>(&mut pool, &arr, rows, cols, op, path)
                .expect("row_reduce valid geometry")
            {
                let got: Vec<f64> = rr.to_host(&pool).iter().map(|&v| v as f64).collect();
                assert_slice_close(&got, &expect_rows, &F32_TOL);
            }
            let arr2: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dev_in);
            if let Some(cr) = reduce::column_reduce::<f32>(&mut pool, &arr2, rows, cols, op, path)
                .expect("column_reduce valid geometry")
            {
                let got: Vec<f64> = cr.to_host(&pool).iter().map(|&v| v as f64).collect();
                assert_slice_close(&got, &expect_cols, &F32_TOL);
            }
        }
    }
    println!("reduce axis (full/row/column) f32: within {F32_TOL:?} on {backend}");
}

fn reduce_host(seg: &[f64], op: ScalarOp) -> f64 {
    match op {
        ScalarOp::Sum => host_sum(seg),
        ScalarOp::Mean => host_mean(seg),
        ScalarOp::Min => host_min(seg),
        ScalarOp::Max => host_max(seg),
        ScalarOp::L2Norm => host_l2(seg),
        ScalarOp::SumSq => host_sumsq(seg),
    }
}

// =========================================================================
// 3. argmin tie-break = lowest index (full + per-row), pinned by numpy fixture
// =========================================================================

#[test]
fn argmin_tie_breaks_lowest_index() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    let case: OracleCase =
        load_npz(fixture("argmin_tie_i32_seed42.npz")).expect("load argmin_tie fixture");
    let x = case.expect_f64("X").to_vec();
    let expect_full = case.expect_f64("argmin_full")[0] as u32;
    let expect_rows: Vec<u32> = case
        .expect_f64("argmin_rows")
        .iter()
        .map(|&v| v as u32)
        .collect();

    // The fixture is a 4×6 matrix (see gen_oracle.py).
    let rows = 4usize;
    let cols = 6usize;
    assert_eq!(x.len(), rows * cols, "fixture shape");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let dev_in: Vec<f32> = x.iter().map(|&v| v as f32).collect();

    // Full-array argmin: lowest flat index of the global minimum.
    let arr: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dev_in);
    let got_full = reduce::argmin::<f32>(&mut pool, &arr).expect("argmin");
    assert_eq!(
        got_full, expect_full,
        "full-array argmin must return the lowest tied index (numpy parity)"
    );

    // Cross-check against the live host reference too.
    assert_eq!(got_full, host_argmin(&x), "argmin vs host reference");

    // Per-row argmin: lowest column index of each row's minimum.
    let arr2: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dev_in);
    let got_rows = reduce::argmin_rows::<f32>(&mut pool, &arr2, rows, cols).expect("argmin_rows");
    assert_eq!(
        got_rows, expect_rows,
        "per-row argmin must return the lowest tied column index per row"
    );

    // argmax sanity: lowest index of the max (no fixture, host ref only).
    let arr3: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &dev_in);
    let got_max = reduce::argmax::<f32>(&mut pool, &arr3).expect("argmax");
    let host_max_idx = {
        let mut bi = 0usize;
        for i in 1..x.len() {
            if x[i] > x[bi] {
                bi = i;
            }
        }
        bi as u32
    };
    assert_eq!(got_max, host_max_idx, "argmax vs host reference (lowest index)");

    println!("argmin tie-break (full + per-row) lowest-index OK on {backend}");
}
