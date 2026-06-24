//! Plan 15-05 — HDBSCAN mutual-reachability GATHER kernel VALUE gate (HDBS-01).
//!
//! The kernel `out[i*n+j] = max(core[i], core[j], d[i*n+j] / alpha)` is the ONLY
//! new device kernel of the phase. This harness launches it under the concrete
//! `ActiveRuntime` (the kernel itself is backend-feature-free in `mlrs-kernels`)
//! and asserts the MR VALUES against an in-test host oracle — incl. a
//! DUPLICATE-POINT row (R-9), the cpu-MLIR silent-miscompile catch (a happy-path
//! non-panic check would miss a mis-lowered GATHER).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate (cpu runs f64; rocm
//! skips per the CubeCL-HIP F64 gap). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::mutual_reachability::mutual_reachability_device;
use mlrs_backend::runtime::{self, ActiveRuntime};

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mr tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("mr tests are f32/f64 only"),
    }
}

/// Host reference: `out[i*n+j] = max(core[i], core[j], d[i*n+j] / alpha)`.
fn mr_reference(d: &[f64], core: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = d[i * n + j] / alpha;
            if core[i] > acc {
                acc = core[i];
            }
            if core[j] > acc {
                acc = core[j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Shared body: launch the kernel over a dense `(n×n)` distance block + per-row
/// core distances, assert each MR value matches the host reference within a tight
/// tolerance (the strict 1e-5 contract; f32 widened to its own epsilon).
fn run_mr_value<F>(d: &[f64], core: &[f64], n: usize, alpha: f64, tol: f64, label: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let d_f: Vec<F> = d.iter().map(|&v| from_f64::<F>(v)).collect();
    let core_f: Vec<F> = core.iter().map(|&v| from_f64::<F>(v)).collect();
    let d_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &d_f);
    let core_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &core_f);

    let mr_dev = mutual_reachability_device::<F>(
        &mut pool,
        &d_dev,
        &core_dev,
        n,
        n,
        from_f64::<F>(alpha),
    )
    .expect("mutual_reachability_device launch");

    let got: Vec<f64> = mr_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let expect = mr_reference(d, core, n, alpha);

    assert_eq!(got.len(), expect.len(), "{label}: MR length mismatch");
    for (idx, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let allclose = abs_err <= tol + tol * e.abs();
        assert!(
            allclose,
            "{label}: MR value mismatch at {idx}: got={g:e} expected={e:e} abs_err={abs_err:e} (tol={tol:e})"
        );
    }
}

/// A small dense distance matrix with a DUPLICATE-POINT row (R-9): points 1 and 2
/// are identical (`d[1][2] = d[2][1] = 0`), so their MR off-diagonal collapses to
/// `max(core[1], core[2])` — the GATHER must read the right `core` entries for the
/// duplicate pair, which a silent cross-index miscompile would get wrong. Distinct
/// non-duplicate distances keep the rest of the matrix discriminating.
fn dup_point_fixture() -> (Vec<f64>, Vec<f64>, usize) {
    let n = 4;
    // Symmetric distance matrix; rows 1 and 2 are a genuine duplicate (d=0).
    let d = vec![
        0.0, 2.0, 2.0, 5.0, //
        2.0, 0.0, 0.0, 4.0, // point 1 == point 2 (distance 0)
        2.0, 0.0, 0.0, 4.0, //
        5.0, 4.0, 4.0, 0.0, //
    ];
    // Per-row core distances (distinct so the three-way max is discriminating).
    let core = vec![1.0, 3.0, 2.5, 0.5];
    (d, core, n)
}

/// `mutual_reachability_value` f32 — the GATHER MR values match the host reference
/// on a duplicate-point fixture (R-9), alpha != 1 exercising the `/alpha` arm.
#[test]
fn mutual_reachability_value_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "dup-point R-9");
    let (d, core, n) = dup_point_fixture();
    run_mr_value::<f32>(&d, &core, n, 2.0, 1e-5, "mutual_reachability_value f32 (alpha=2)");
    // alpha = 1 (no scaling) — the plain three-way max.
    run_mr_value::<f32>(&d, &core, n, 1.0, 1e-5, "mutual_reachability_value f32 (alpha=1)");
}

/// `mutual_reachability_value` f64 — the GATHER MR values on the duplicate-point
/// fixture (cpu runs f64; rocm skips-with-log).
#[test]
fn mutual_reachability_value_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "dup-point R-9");
    if capability::skip_f64_with_log() {
        println!("mutual_reachability_value f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let (d, core, n) = dup_point_fixture();
    run_mr_value::<f64>(&d, &core, n, 2.0, 1e-12, "mutual_reachability_value f64 (alpha=2)");
    run_mr_value::<f64>(&d, &core, n, 1.0, 1e-12, "mutual_reachability_value f64 (alpha=1)");

    // The duplicate pair (1,2) must collapse to max(core[1], core[2]) off-diagonal
    // (d=0/alpha=0), an explicit R-9 value assertion beyond the bulk compare.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let d_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &d);
    let core_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &core);
    let mr = mutual_reachability_device::<f64>(&mut pool, &d_dev, &core_dev, n, n, 1.0)
        .expect("dup MR launch");
    let got = mr.to_host(&pool);
    let off = got[1 * n + 2];
    assert!(
        (off - core[1].max(core[2])).abs() < 1e-12,
        "R-9: duplicate pair (1,2) MR must be max(core1,core2)={}, got {off}",
        core[1].max(core[2])
    );
}
