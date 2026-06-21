//! Plan 09-02 — laplacian (PRIM-09) standalone validation.
//!
//! These were the Wave-0 (09-01) `#[ignore]` Nyquist scaffolds; plan 09-02
//! un-ignores them and wires the real `laplacian` compute against the committed
//! `.npz` oracle fixtures:
//!
//!   - `laplacian_value` — `L = I − D^-1/2 A D^-1/2` and `dd = sqrt(degree)` vs
//!     the scipy `_laplacian_dense` host reference, f32 (documented band) + f64
//!     (strict `F64_TOL`).
//!   - `zero_degree` — an isolated-node fixture produces NO NaN / NO infinite
//!     value: the zero-degree node's `dd == 1`, its `L` row is all-zero, and its
//!     `L` diagonal is `0` (the typed-zero guard success criterion).
//!   - `memory_gate` — `BufferPool` / `PoolStats` reuse-bounded gate: driving
//!     `laplacian` N× at a fixed shape conserves `live_bytes` and plateaus
//!     `peak_bytes` after warmup, and performs ZERO mid-pipeline metered
//!     read-backs (mirrors `memory_gate_test.rs`).
//!
//! f64 carries the `skip_f64_with_log` capability gate verbatim (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-backend/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, load_npz, OracleCase, Tolerance, F64_TOL};

/// laplacian fixture geometry (gen_oracle.py `LAP_N` × `LAP_N`).
const N: usize = 8;

/// Documented f32 band for the PRIM-09 Laplacian (set FROM the measurement
/// printed by the value test). f64 stays strict `F64_TOL` (1e-5). The Laplacian
/// is a divide-by-`sqrt(degree)` map over a finite affinity; the band matches the
/// Phase-8 kernel-matrix f32 precedent.
const LAP_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Build an `F` (f32/f64) from an `f64` (mirrors kernel_matrix_test::from_f64).
fn from_f64<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("laplacian is f32/f64 only"),
    }
}

/// Reinterpret an `F` value back to `f64` for the oracle comparison.
fn to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!(),
    }
}

/// Assert the fixture exposes a well-formed `n×n` affinity + reference Laplacian
/// + length-n degree-normalization vector.
fn assert_shapes(case: &OracleCase, n: usize) {
    assert_eq!(case.expect_f64("A").len(), n * n, "A must be n×n");
    assert_eq!(case.expect_f64("L").len(), n * n, "L must be n×n");
    assert_eq!(case.expect_f64("dd").len(), n, "dd must be length n");
}

/// Run `laplacian` on the fixture's `A` at precision `F`, returning `(L, dd)`
/// read back to host `f64`.
fn compute_laplacian<F>(case: &OracleCase) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let a_f: Vec<F> = case.expect_f64("A").iter().map(|&v| from_f64::<F>(v)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a_f);

    let (l, dd) = laplacian::<F>(&mut pool, &a_dev, N).expect("laplacian computes");

    let l_host: Vec<f64> = l.to_host(&pool).iter().map(|&v| to_f64::<F>(v)).collect();
    let dd_host: Vec<f64> = dd.to_host(&pool).iter().map(|&v| to_f64::<F>(v)).collect();
    (l_host, dd_host)
}

/// Drive `laplacian` at precision `F` and assert `L` and `dd` against the
/// committed scipy `_laplacian_dense` reference within `tol`.
fn run_value<F>(case: &OracleCase, tol: &Tolerance, dtype: &str)
where
    F: Float + CubeElement + Pod,
{
    let (l_got, dd_got) = compute_laplacian::<F>(case);
    let l_exp = case.expect_f64("L");
    let dd_exp = case.expect_f64("dd");

    let mut max_abs = 0.0f64;
    for (g, &e) in l_got.iter().zip(l_exp.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    let mut dd_max_abs = 0.0f64;
    for (g, &e) in dd_got.iter().zip(dd_exp.iter()) {
        dd_max_abs = dd_max_abs.max((g - e).abs());
    }
    println!(
        "laplacian[{dtype}] L max_abs={max_abs:e} dd max_abs={dd_max_abs:e} \
         (tol.abs={:e} tol.rel={:e})",
        tol.abs, tol.rel
    );

    assert_slice_close(&l_got, l_exp, tol);
    assert_slice_close(&dd_got, dd_exp, tol);
}

/// PRIM-09 normalized Laplacian vs scipy `_laplacian_dense`, f64 strict. Gated by
/// `skip_f64_with_log`.
#[test]
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
    run_value::<f64>(&case, &F64_TOL, "f64");
}

/// PRIM-09 normalized Laplacian vs scipy `_laplacian_dense`, f32 documented band.
/// Runs on every backend (the f32 gate is rocm; cpu also exercises f32).
#[test]
fn laplacian_value_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("laplacian_f32_seed42.npz")).expect("load laplacian_f32");
    assert_shapes(&case, N);
    run_value::<f32>(&case, &LAP_F32_BAND, "f32");
}

/// PRIM-09 no-NaN / no-infinite-value on a zero-degree (isolated-node) graph —
/// the typed-zero guard success criterion. The isolated node's `dd == 1`, its `L`
/// row is all-zero, and its `L` diagonal is `0`.
#[test]
fn zero_degree() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("laplacian zero_degree f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("laplacian_isolated_f64_seed42.npz"))
        .expect("load laplacian_isolated_f64");
    assert_shapes(&case, N);

    // Every reference Laplacian entry is finite (sanity on the committed blob).
    for &v in case.expect_f64("L") {
        assert!(v.is_finite(), "reference L must be finite on isolated nodes");
    }

    let (l_got, dd_got) = compute_laplacian::<f64>(&case);

    // 1. The DEVICE output is finite everywhere — the typed-zero guard never emits
    //    a NaN or an infinite value on the zero-degree node (T-9-LAP).
    for (idx, &v) in l_got.iter().enumerate() {
        assert!(
            v.is_finite(),
            "device L[{idx}] = {v} is not finite — the zero-degree guard produced a \
             NaN / infinite value (T-9-LAP regression)"
        );
    }
    for (idx, &v) in dd_got.iter().enumerate() {
        assert!(dd_got[idx].is_finite(), "device dd[{idx}] = {v} is not finite");
    }

    // 2. Identify the isolated node from the reference dd (dd == 1 AND its A row is
    //    all-zero). The committed fixture's isolated node is the last one.
    let a = case.expect_f64("A");
    for i in 0..N {
        let row_sum: f64 = (0..N).map(|j| a[i * N + j].abs()).sum();
        if row_sum == 0.0 {
            // Isolated node: dd == 1, L diagonal == 0, L row all-zero.
            assert!(
                (dd_got[i] - 1.0).abs() < 1e-12,
                "isolated node {i}: dd={} expected 1 (typed-zero guard)",
                dd_got[i]
            );
            assert!(
                l_got[i * N + i].abs() < 1e-12,
                "isolated node {i}: L diagonal={} expected 0 (1 - isolated)",
                l_got[i * N + i]
            );
            for j in 0..N {
                assert!(
                    l_got[i * N + j].abs() < 1e-12,
                    "isolated node {i}: L[{i},{j}]={} expected 0 (whole row zero)",
                    l_got[i * N + j]
                );
            }
        }
    }

    // 3. The device output value-matches the reference (strict f64) too.
    assert_slice_close(&l_got, case.expect_f64("L"), &F64_TOL);
    assert_slice_close(&dd_got, case.expect_f64("dd"), &F64_TOL);
}

/// PRIM-09 `BufferPool` / `PoolStats` reuse-bounded memory gate (mirrors
/// `memory_gate_test.rs`): driving `laplacian` N× at a fixed n×n shape conserves
/// `live_bytes` and plateaus `peak_bytes` after warmup (the transient
/// diagonal-zeroed working buffer + degree vector are released each call), and
/// performs ZERO mid-pipeline metered read-backs (the whole prim stays
/// device-resident — the only `read_backs` would be a caller's terminal compare,
/// which this gate never issues).
#[test]
fn memory_gate() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    const ITERS: usize = 5;
    let n = N;

    // Deterministic finite affinity (the gate asserts on POOL COUNTERS, not on
    // oracle values; any reproducible non-negative fill suffices).
    let make = |seed: usize| -> Vec<f32> {
        (0..n * n)
            .map(|i| (((i + seed) % 13) as f32) * 0.1)
            .collect()
    };

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let mut live_after: Vec<u64> = Vec::with_capacity(ITERS);
    let mut peak_after: Vec<u64> = Vec::with_capacity(ITERS);

    for iter in 0..ITERS {
        let a = make(iter);
        let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);

        let (l, dd) = laplacian::<f32>(&mut pool, &a_dev, n).expect("laplacian in memory gate");

        // Release this call's transient operands + the produced (L, dd) so the
        // steady-state footprint conserves (the prim releases its own internal
        // working buffer + degree scratch; the caller owns L/dd each iteration).
        a_dev.release_into(&mut pool);
        l.release_into(&mut pool);
        dd.release_into(&mut pool);

        let stats = pool.stats();
        live_after.push(stats.live_bytes);
        peak_after.push(stats.peak_bytes);
    }

    // No mid-pipeline metered read-back across the whole gate: the prim is
    // device-resident end-to-end (its internal row_reduce uses plain `to_host`,
    // which deliberately does NOT bump the metered `read_backs` counter), and the
    // gate itself never issues a `to_host_metered`.
    assert_eq!(
        pool.stats().read_backs,
        0,
        "laplacian memory_gate FAILED on {backend}: read_backs={} (expected 0) — the \
         prim performed a metered mid-pipeline device→host round-trip. stats={:?}",
        pool.stats().read_backs,
        pool.stats()
    );

    // After a warmup iteration the live footprint must CONSERVE: the prim releases
    // its diagonal-zeroed working buffer and the degree vector. A monotone climb is
    // the RED-if-removed signal that a release went missing (build-failing).
    for w in 2..ITERS {
        assert!(
            live_after[w] <= live_after[1],
            "laplacian memory_gate FAILED on {backend}: live_bytes grows after warmup \
             (iter {w} = {} > iter 1 = {}) — a transient release went missing",
            live_after[w],
            live_after[1]
        );
    }
    // peak_bytes plateaus after warmup (released scratch reused in place).
    for w in 2..ITERS {
        assert_eq!(
            peak_after[w], peak_after[ITERS - 1],
            "laplacian memory_gate FAILED on {backend}: peak_bytes must plateau after \
             warmup (iter {w} vs final)"
        );
    }

    println!(
        "laplacian memory_gate backend={backend}: live={live_after:?} peak={peak_after:?} \
         read_backs={} (ITERS={ITERS})",
        pool.stats().read_backs
    );
}
