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
fn saxpy_runs_on_active_backend() {
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
