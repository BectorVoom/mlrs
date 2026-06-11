//! Wave-0 toolchain/API spike — live integration tests.
//!
//! This test file OWNS a concrete runtime feature (the kernels crate is
//! feature-free, so the live launch proof must live here). It is the
//! executable evidence behind `SPIKE-FINDINGS.md`:
//!   - `saxpy_runs_on_active_backend` (Task 2): the generic `<F: Float>` saxpy
//!     kernel launches on the active runtime for f32 and matches a host
//!     reference exactly.
//!
//! Task 3 extends this file with capability / `Bytes` / npz probes.
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests` in
//! `src/`.

use cubecl::prelude::*;
use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_kernels::saxpy_kernel;

/// Standard ceiling-division 1D launch config helper.
fn launch_dims(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

#[test]
fn spike_saxpy_runs_on_active_backend() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();

    // Integer-valued inputs so the f32 result matches the host reference
    // exactly (no rounding slack needed for this smoke proof).
    let a: f32 = 3.0;
    let x_host: Vec<f32> = (0..1024).map(|i| (i % 7) as f32).collect();
    let y_host: Vec<f32> = (0..1024).map(|i| (i % 5) as f32).collect();
    let n = x_host.len();

    // Host reference: y = a*x + y.
    let expected: Vec<f32> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(&xi, &yi)| a * xi + yi)
        .collect();

    // Upload. (Task 3 records the exact `Bytes` constructor + copy semantics.)
    let x_handle = client.create(cubecl::bytes::Bytes::from_elems(x_host));
    let y_handle = client.create(cubecl::bytes::Bytes::from_elems(y_host));

    let (count, dim) = launch_dims(n);

    // `read_one` consumes a `Handle`; clone the output handle before the launch
    // also consumes one (CubeCL handles are cheap ref-counted clones).
    let y_read = y_handle.clone();

    saxpy_kernel::launch::<f32, ActiveRuntime>(
        &client,
        count,
        dim,
        // A6 NOTE: in cubecl 0.10 a scalar kernel arg is passed by value
        // directly (no `ScalarArg` wrapper) in the generated launch fn.
        a,
        // SAFETY: length is derived from the validated host slice `.len()`; the
        // kernel bounds-checks `if tid < x.len()` (mitigates T-01-01).
        // `from_raw_parts(handle, len)` — 2 args, takes Handle by value.
        unsafe { ArrayArg::from_raw_parts(x_handle, n) },
        unsafe { ArrayArg::from_raw_parts(y_handle, n) },
    );

    let bytes = client.read_one(y_read).expect("read-back of y handle");
    let got: &[f32] = bytemuck::cast_slice(&bytes);

    assert_eq!(got.len(), n, "read-back length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "saxpy mismatch at index {i}: got {g}, expected {e}");
    }
}

/// Task 3 / A1 + A2: the capability query compiles and returns a value on the
/// active backend; log the f32/f64 support so the dtype/backend line shows up
/// in CI (Criterion 4).
#[test]
fn spike_capability_query_reports_f64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();

    // f32 is universally supported and must report true everywhere.
    let f32_ok = capability::supports_type(&client, FloatKind::F32);
    let f64_ok = capability::supports_f64(&client);

    let backend = if cfg!(feature = "cpu") {
        "cpu"
    } else if cfg!(feature = "wgpu") {
        "wgpu"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "rocm"
    };

    log::info!("capability backend={backend} f32_supported={f32_ok} f64_supported={f64_ok}");
    println!("capability backend={backend} f32_supported={f32_ok} f64_supported={f64_ok}");

    assert!(f32_ok, "f32 must be supported on every backend ({backend})");
    // f64 is adapter-dependent: we only assert the query RETURNS a value
    // (above), not its truthiness. The active-runtime facade must agree.
    assert_eq!(
        capability::feature_enabled(FloatKind::F64),
        f64_ok,
        "feature_enabled facade must match supports_f64 on the active client"
    );
}

/// Task 3 / A3: probe the `cubecl::bytes::Bytes` constructors to determine the
/// host-copy semantics for the Arrow bridge (Plan 03).
#[test]
fn spike_bytes_constructor_semantics() {
    let client = runtime::active_client();

    // `from_elems(Vec<T>)` takes ownership of a typed Vec (used by saxpy above).
    let from_elems = cubecl::bytes::Bytes::from_elems(vec![1.0f32, 2.0, 3.0, 4.0]);
    let h1 = client.create(from_elems);
    let r1 = client.read_one(h1).expect("read from_elems");
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&r1), &[1.0, 2.0, 3.0, 4.0]);

    // `from_bytes_vec(Vec<u8>)` takes an owned byte Vec. The manuals call this
    // with `slice.to_vec()` — i.e. a HOST COPY. A3 RESOLVED: both 0.10
    // constructors consume an owned allocation, so the Arrow handoff is
    // "validated single-upload", not literal host zero-copy (recorded in
    // SPIKE-FINDINGS.md; Plan 03 documents the honest semantics).
    let raw: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&[5.0f32, 6.0]).to_vec();
    let from_bytes = cubecl::bytes::Bytes::from_bytes_vec(raw);
    let h2 = client.create(from_bytes);
    let r2 = client.read_one(h2).expect("read from_bytes_vec");
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&r2), &[5.0, 6.0]);
}

/// Task 3 / A7: confirm `bytemuck::try_cast_slice` surfaces misalignment as a
/// recoverable `Err` (not a panic), so the Arrow bridge (Plan 03) can map it to
/// a typed `BridgeError::Misaligned`.
#[test]
fn spike_try_cast_slice_is_recoverable() {
    // Well-aligned f32 slice casts to u8 cleanly.
    let aligned = [1.0f32, 2.0, 3.0];
    assert!(bytemuck::try_cast_slice::<f32, u8>(&aligned).is_ok());

    // Force misalignment: take a u8 buffer and try to view it as f32 starting
    // at an odd offset. try_cast_slice returns Err (alignment), never panics.
    let raw = [0u8; 16];
    let misaligned = &raw[1..13]; // start offset 1 => not 4-byte aligned for f32
    let res = bytemuck::try_cast_slice::<u8, f32>(misaligned);
    assert!(
        res.is_err(),
        "A7: expected recoverable Err on misaligned cast, got Ok"
    );
}

/// Task 3 / A4: full npz round-trip through `npyz` (read + write), no numpy
/// required. Proves named-array `by_name` access works for f32 and f64 — the
/// exact API the oracle loader (Plan 02) will use.
#[test]
fn spike_npz_named_array_round_trip() {
    use npyz::npz::{NpzArchive, NpzWriter};
    use npyz::WriterBuilder;
    use std::io::Cursor;

    let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let coef_f64: Vec<f64> = vec![0.5, -1.25, 3.0];

    // Write a throwaway in-memory .npz with two named arrays of different dtype.
    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut npz = NpzWriter::new(&mut buf);
        npz.array("x", Default::default())
            .unwrap()
            .default_dtype()
            .shape(&[x_f32.len() as u64])
            .begin_nd()
            .unwrap()
            .extend(x_f32.iter().copied())
            .unwrap();
        npz.array("coef_", Default::default())
            .unwrap()
            .default_dtype()
            .shape(&[coef_f64.len() as u64])
            .begin_nd()
            .unwrap()
            .extend(coef_f64.iter().copied())
            .unwrap();
    }

    // Read it back by name (A4 API proof).
    let bytes = buf.into_inner();
    let mut npz = NpzArchive::new(Cursor::new(&bytes[..])).expect("open npz");

    let mut names: Vec<String> = npz.array_names().map(|s| s.to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["coef_".to_string(), "x".to_string()]);

    let x_read: Vec<f32> = npz
        .by_name("x")
        .expect("by_name x")
        .expect("x present")
        .into_vec::<f32>()
        .expect("decode f32");
    assert_eq!(x_read, x_f32);

    let coef_read: Vec<f64> = npz
        .by_name("coef_")
        .expect("by_name coef_")
        .expect("coef_ present")
        .into_vec::<f64>()
        .expect("decode f64");
    assert_eq!(coef_read, coef_f64);
}
