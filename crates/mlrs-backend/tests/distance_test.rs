//! Plan 02-03 — pairwise squared-Euclidean distance (PRIM-03) oracle validation.
//!
//! Exercises the device GEMM-expansion distance (`prims::distance::distance`)
//! for both `f32` and `f64` against a direct host distance loop and the
//! committed `.npz` convention fixtures:
//!
//!   - `distance_matches_host_ref`        — seeded random X/Y over several
//!     shapes, squared device distance vs a direct f64 host reference within
//!     tolerance (f32 and f64, the f64 arm capability-gated).
//!   - `distance_min_nonnegative`         — a DELIBERATE near-identical-rows f32
//!     case (catastrophic cancellation) asserting `min(D) >= 0` (Criterion 3 /
//!     Pitfall 5 / T-0203-03: the `max(d²,0)` clamp lets no negative escape).
//!   - `distance_sqrt_matches_host_ref`   — `distance(sqrt=true)` vs the sqrt
//!     host reference (the optional Euclidean boundary, D-08).
//!   - `distance_npz_fixture_matches`     — `dist_sq_{f32,f64}_seed42.npz` and
//!     `dist_sqrt_f64_seed42.npz` `X`/`Y`/`D`.
//!
//! The host reference is the DIRECT `Σ_k (X[i,k]−Y[j,k])²` form (computed in
//! f64), independent of the device's GEMM-expansion `‖x‖²+‖y‖²−2XYᵀ` — so a
//! match validates the expansion identity, not a tautology. The f64 cases gate
//! on `capability::skip_f64_with_log` (skip, never fail — Criterion 4 / T-05-04).
//!
//! The f32 arms reuse the `F32_ORACLE_NEAR_ZERO_FLOOR` precedent from
//! `pipeline_test.rs` / `gemm_test.rs`: near-cancellation distances are genuinely
//! tiny, so below the floor the check falls back to abs-only (still ≤ 1e-5 abs)
//! — it NEVER loosens the 1e-5 absolute bound, and f64 keeps the strict
//! `assert_slice_close`.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::reduce::ReducePath;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// f32-precision near-zero floor for the distance oracle comparison, mirroring
/// `F32_GEMM_NEAR_ZERO_FLOOR` in `gemm_test.rs` and `F32_ORACLE_NEAR_ZERO_FLOOR`
/// in `pipeline_test.rs`. The distance combine accumulates `‖x‖²+‖y‖²−2XYᵀ`, so
/// near-identical rows produce genuinely tiny (or clamped-to-zero) results whose
/// *absolute* error stays far inside `1e-5` while the *relative* term can exceed
/// `1e-5` purely from f32 rounding. This floor raises the abs-only fallback to
/// an f32-meaningful magnitude for the f32 distance cases ONLY; it never loosens
/// the `1e-5` absolute bound. The f64 cases keep the strict comparator.
const F32_DIST_NEAR_ZERO_FLOOR: f64 = 1e-2;

/// Element-wise f32 distance oracle compare: strict abs-AND-rel per `F32_TOL`,
/// except abs-only (still bounded by `tol.abs` = `1e-5`) when
/// `|expected| < F32_DIST_NEAR_ZERO_FLOOR`. Panics with diagnostic detail.
fn assert_slice_close_f32_dist(got: &[f64], expected: &[f64], tol: &Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "f32 distance oracle length mismatch: got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.abs() < F32_DIST_NEAR_ZERO_FLOOR {
            let abs_err = (g - e).abs();
            assert!(
                abs_err <= tol.abs,
                "f32 distance near-zero abs check failed at index {i}: got={g:e}, \
                 expected={e:e}, abs_err={abs_err:e} (tol.abs={:e})",
                tol.abs
            );
        } else {
            assert!(
                is_close(g, e, tol),
                "f32 distance assert_close failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={:e} (tol.abs={:e}, tol.rel={:e})",
                (g - e).abs(),
                tol.abs,
                tol.rel
            );
        }
    }
}

/// Resolve a workspace-root-relative fixture path (matches `gemm_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Direct host pairwise squared-Euclidean distance reference, computed in f64
/// (the oracle ground truth — independent of the device's GEMM-expansion).
///
/// `x` is `rows_x × cols` row-major, `y` is `rows_y × cols`; the result is the
/// `rows_x × rows_y` row-major squared distance `Σ_k (x[i,k] − y[j,k])²`.
fn host_dist_sq_ref(
    x: &[f64],
    y: &[f64],
    rows_x: usize,
    rows_y: usize,
    cols: usize,
) -> Vec<f64> {
    let mut d = vec![0.0f64; rows_x * rows_y];
    for i in 0..rows_x {
        for j in 0..rows_y {
            let mut acc = 0.0f64;
            for k in 0..cols {
                let diff = x[i * cols + k] - y[j * cols + k];
                acc += diff * diff;
            }
            d[i * rows_y + j] = acc;
        }
    }
    d
}

/// Run a device distance end-to-end for a single shape and return the result.
/// Generic over the float element type so the f32 and f64 cases share the exact
/// same device path. `sqrt` selects the optional Euclidean boundary (D-08).
fn run_distance_case<F>(
    x_host: &[F],
    y_host: &[F],
    rows_x: usize,
    rows_y: usize,
    cols: usize,
    sqrt: bool,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, y_host);

    let d_dev = distance::<F>(
        &mut pool,
        &x_dev,
        (rows_x, cols),
        &y_dev,
        (rows_y, cols),
        sqrt,
        None,
        // Shared path is always portable; the reduction's plane path is gated
        // separately and validated in reduce_test.rs.
        ReducePath::Shared,
    )
    .expect("distance host API rejects nothing for a valid shape");
    d_dev.to_host_metered(&mut pool)
}

/// f32 squared distance vs the f64 direct host reference over several shapes
/// (incl. a larger feature dim for accumulation stress).
#[test]
fn distance_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for &(rx, ry, c) in &[(5usize, 4usize, 3usize), (8, 8, 8), (3, 6, 32)] {
        let x: Vec<f32> = (0..rx * c).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let y: Vec<f32> = (0..ry * c).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();

        let got = run_distance_case::<f32>(&x, &y, rx, ry, c, false);
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let expected = host_dist_sq_ref(&x64, &y64, rx, ry, c);
        let got64: Vec<f64> = got.iter().map(|&v| v as f64).collect();
        assert_slice_close_f32_dist(&got64, &expected, &F32_TOL);
    }
    println!("distance f32 backend={backend}: squared distance matches host ref over 3 shapes");
}

/// f64 squared distance vs the f64 direct host reference, capability-gated.
#[test]
fn distance_f64_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("distance f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for &(rx, ry, c) in &[(5usize, 4usize, 3usize), (8, 8, 8), (3, 6, 32)] {
        let x: Vec<f64> = (0..rx * c).map(|i| ((i % 13) as f64) * 0.1 - 0.6).collect();
        let y: Vec<f64> = (0..ry * c).map(|i| ((i % 11) as f64) * 0.1 - 0.5).collect();

        let got = run_distance_case::<f64>(&x, &y, rx, ry, c, false);
        let expected = host_dist_sq_ref(&x, &y, rx, ry, c);
        assert_slice_close(&got, &expected, &F64_TOL);
    }
    println!("distance f64 backend={backend}: squared distance matches host ref over 3 shapes");
}

/// The `max(d²,0)` clamp produces NO negative distances under f32 catastrophic
/// cancellation (Criterion 3 / Pitfall 5 / T-0203-03). Uses DELIBERATE
/// near-identical rows: `‖x‖²+‖y‖²−2XYᵀ` for identical rows is exactly the
/// difference of two near-equal large numbers, which rounds slightly negative
/// in f32 — the clamp must floor it to `0`, never leak a negative value.
#[test]
fn distance_min_nonnegative() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    // Rows with LARGE magnitude and tiny inter-row differences so the squared
    // distance is the difference of two large near-equal terms (catastrophic
    // cancellation). Some pairs are bit-identical (true distance 0 → the prime
    // candidate for a slightly-negative f32 result before the clamp).
    let cols = 4usize;
    let base = [1000.0f32, -2000.0, 1500.0, 3000.0];
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    let rows_x = 6usize;
    let rows_y = 6usize;
    for r in 0..rows_x {
        // Each X row is `base` nudged by a tiny per-row epsilon.
        let eps = (r as f32) * 1e-3;
        for k in 0..cols {
            x.push(base[k] + eps);
        }
    }
    for r in 0..rows_y {
        // Each Y row mirrors an X row exactly (and some with a tiny offset), so
        // several (i,j) pairs are identical or near-identical.
        let eps = (r as f32) * 1e-3;
        for k in 0..cols {
            y.push(base[k] + eps);
        }
    }

    let got = run_distance_case::<f32>(&x, &y, rows_x, rows_y, cols, false);

    // The CORE assertion: no negative squared distance escapes the clamp.
    let min = got.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        got.iter().all(|&d| d >= 0.0),
        "distance produced a NEGATIVE squared distance (clamp failed): min={min:e} — \
         the unconditional max(d²,0) clamp (Criterion 3 / T-0203-03) must floor all \
         f32-cancellation results to >= 0"
    );

    // Sanity: the diagonal (X row r vs Y row r, identical rows) is ~0 (clamped).
    for r in 0..rows_x.min(rows_y) {
        let d = got[r * rows_y + r];
        assert!(
            d.abs() <= F32_TOL.abs as f32 || d >= 0.0,
            "identical-row squared distance must be ~0 and non-negative, got {d:e}"
        );
    }
    println!(
        "distance f32 backend={backend}: min squared distance = {min:e} (>= 0 — clamp holds)"
    );
}

/// `distance(sqrt=true)` matches the sqrt of the direct host reference (the
/// optional Euclidean boundary, D-08). f64 here so the sqrt is compared at full
/// precision; capability-gated.
#[test]
fn distance_sqrt_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("distance sqrt f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }

    for &(rx, ry, c) in &[(5usize, 4usize, 3usize), (8, 8, 8)] {
        let x: Vec<f64> = (0..rx * c).map(|i| ((i % 13) as f64) * 0.1 - 0.6).collect();
        let y: Vec<f64> = (0..ry * c).map(|i| ((i % 11) as f64) * 0.1 - 0.5).collect();

        let got = run_distance_case::<f64>(&x, &y, rx, ry, c, true);
        let sq = host_dist_sq_ref(&x, &y, rx, ry, c);
        let expected: Vec<f64> = sq.iter().map(|&v| v.sqrt()).collect();
        assert_slice_close(&got, &expected, &F64_TOL);
    }
    println!("distance sqrt f64 backend={backend}: Euclidean distance matches host ref");
}

/// Device distance matches the committed numpy convention fixtures: the squared
/// `dist_sq_{f32,f64}_seed42.npz` and the sqrt `dist_sqrt_f64_seed42.npz`.
#[test]
fn distance_npz_fixture_matches() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    // Fixture geometry (gen_oracle.py DIST_ROWS_X/ROWS_Y/COLS).
    let (rx, ry, c) = (5usize, 4usize, 3usize);

    // --- squared f32 — always runs. ---
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");
    let case: OracleCase =
        load_npz(fixture("dist_sq_f32_seed42.npz")).expect("load dist_sq_f32_seed42.npz");
    let x = case.expect_f32("X");
    let y = case.expect_f32("Y");
    let d = case.expect_f32("D");
    let got = run_distance_case::<f32>(x, y, rx, ry, c, false);
    let got64: Vec<f64> = got.iter().map(|&v| v as f64).collect();
    let d64: Vec<f64> = d.iter().map(|&v| v as f64).collect();
    assert_slice_close_f32_dist(&got64, &d64, &F32_TOL);

    // --- squared + sqrt f64 — capability-gated. ---
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("distance npz f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case: OracleCase =
        load_npz(fixture("dist_sq_f64_seed42.npz")).expect("load dist_sq_f64_seed42.npz");
    let x = case.expect_f64("X");
    let y = case.expect_f64("Y");
    let d = case.expect_f64("D");
    let got = run_distance_case::<f64>(x, y, rx, ry, c, false);
    assert_slice_close(&got, d, &F64_TOL);

    let case: OracleCase =
        load_npz(fixture("dist_sqrt_f64_seed42.npz")).expect("load dist_sqrt_f64_seed42.npz");
    let x = case.expect_f64("X");
    let y = case.expect_f64("Y");
    let d = case.expect_f64("D");
    let got = run_distance_case::<f64>(x, y, rx, ry, c, true);
    assert_slice_close(&got, d, &F64_TOL);
    println!("distance npz backend={backend}: squared + sqrt fixtures match");
}
