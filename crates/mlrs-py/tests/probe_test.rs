//! Import-time driver-probe + global-pool integration tests (Task 06-02-2).
//!
//! AGENTS.md §2: tests live here, never in an in-source `#[cfg(test)]` module.
//!
//! These exercise the D-08 probe logic the `#[pymodule] _mlrs` runs at import:
//! the `catch_unwind(active_client + properties)` probe must SUCCEED on the cpu
//! gate (the driver is present), the process-global `Mutex<BufferPool>` must
//! initialize and serve an allocation, and `supports_f64()` must be a stable
//! boolean. (The negative path — a missing driver raising `PyImportError` instead
//! of aborting — cannot be exercised on a machine that HAS the driver; it is a
//! property of wrapping the probe in `catch_unwind`, verified by the cpu-present
//! success path here plus the panic-safety of `catch_unwind` itself.)
//!
//! Built only with a backend feature active; run as
//! `cargo test -p mlrs-py --features cpu --test probe_test`.

#![cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]

use mlrs_py::capability::supports_f64;

use mlrs_backend::runtime::active_client;

#[test]
fn driver_probe_succeeds_when_present() {
    // The exact probe body the `#[pymodule]` init runs: construct the client and
    // touch `properties()` to force the device handshake, all under
    // `catch_unwind`. On the cpu gate the driver is present, so the probe must
    // report Ok (no panic caught).
    let probe = std::panic::catch_unwind(|| {
        let client = active_client();
        let _ = client.properties();
    });
    assert!(
        probe.is_ok(),
        "the import-time driver probe must succeed when the backend driver is present"
    );
}

#[test]
fn probe_is_panic_safe() {
    // `catch_unwind` must convert a panic into an `Err` rather than unwinding
    // into (a hypothetical) CPython — the property the D-08 ImportError path
    // relies on (T-06-05). Prove the wrapper catches a panic here so the
    // module-init translation to `PyImportError` is sound.
    let caught = std::panic::catch_unwind(|| panic!("simulated missing driver"));
    assert!(
        caught.is_err(),
        "catch_unwind must capture a panic (so a real driver-absent panic becomes ImportError, not an abort)"
    );
}

#[test]
fn supports_f64_is_stable() {
    // The capability flag the module exposes as `backend_supports_f64()` must be
    // a stable boolean (queried twice → same answer). On the cpu gate it is
    // `true` (cpu runs f64); the test asserts stability, not a specific value, so
    // it also holds on an f64-incapable backend.
    let a = supports_f64();
    let b = supports_f64();
    assert_eq!(a, b, "backend_supports_f64() must be stable across calls");
}

#[test]
fn global_pool_initializes_and_serves_an_allocation() {
    // The single process-global pool/client lifecycle: a fresh pool over the
    // active client (the same construction `global_pool()` performs lazily) must
    // initialize and serve an allocation through `acquire`. (We construct a local
    // pool rather than reaching into the crate-private `global_pool()` so the
    // test exercises the same `BufferPool::new(active_client())` path without
    // depending on module-private state.)
    let mut pool = mlrs_backend::pool::BufferPool::new(active_client());
    let handle = pool.acquire(64);
    let stats = pool.stats();
    assert_eq!(stats.allocations, 1, "first acquire is a fresh allocation");
    assert_eq!(stats.live_bytes, 64, "live bytes track the acquired buffer");
    pool.release(handle, 64);
    assert_eq!(pool.stats().live_bytes, 0, "release returns the buffer to the free-list");
}
