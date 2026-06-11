//! Plan 04 — memory-efficiency layer integration tests.
//!
//! Two concerns are proven here against the active runtime (cpu / wgpu):
//!   - `BufferPool` free-list reuse + logged-only `PoolStats` counters
//!     (FOUND-05 / D-04 / D-05). Per D-05 the counters are inspected for
//!     correctness of the API, NOT used as a hard reuse-rate phase gate — the
//!     trivial Phase-1 workloads do not exercise realistic allocation yet.
//!   - `DeviceArray<R, F>` host↔device round-trip with pool-routed allocation
//!     (FOUND-05).
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests` in
//! `src/`.

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::{BufferPool, PoolStats};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// acquire on an empty pool calls `client.empty` and bumps `allocations`;
/// a subsequent acquire of the SAME byte size after a release reuses the freed
/// handle and bumps `reuses` (NOT `allocations`). live/peak bytes track usage.
#[test]
fn pool_reuses_released_buffer_of_matching_size() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let size = 256usize;

    // Fresh acquire on an empty pool: allocation, not reuse.
    let h0 = pool.acquire(size);
    assert_eq!(pool.stats().allocations, 1, "first acquire allocates");
    assert_eq!(pool.stats().reuses, 0, "first acquire is not a reuse");
    assert_eq!(pool.stats().live_bytes, size as u64, "live tracks the acquire");
    assert_eq!(pool.stats().peak_bytes, size as u64, "peak tracks the acquire");

    // Release returns the handle to the free-list; live drops.
    pool.release(h0, size);
    assert_eq!(pool.stats().live_bytes, 0, "release drops live_bytes");
    assert_eq!(
        pool.stats().peak_bytes,
        size as u64,
        "peak is a high-water mark, not reduced by release"
    );

    // Second acquire of the same size REUSES; allocations unchanged.
    let _h1 = pool.acquire(size);
    assert_eq!(pool.stats().reuses, 1, "second acquire of same size reuses");
    assert_eq!(
        pool.stats().allocations,
        1,
        "reuse must NOT increment allocations"
    );
    assert_eq!(pool.stats().live_bytes, size as u64, "live back up after reuse");
}

/// A different byte size has no free entry and must allocate fresh.
#[test]
fn pool_distinct_sizes_do_not_alias() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let a = pool.acquire(128);
    pool.release(a, 128);
    // Different size: no reusable entry, must allocate.
    let _b = pool.acquire(256);
    assert_eq!(pool.stats().allocations, 2, "distinct sizes both allocate");
    assert_eq!(pool.stats().reuses, 0, "no reuse across distinct sizes");
}

/// `PoolStats` exposes the four required counters and is `Debug` (logged-only).
#[test]
fn pool_stats_exposes_counters_and_logs() {
    let _ = env_logger::builder().is_test(true).try_init();

    let stats = PoolStats::default();
    assert_eq!(stats.allocations, 0);
    assert_eq!(stats.reuses, 0);
    assert_eq!(stats.peak_bytes, 0);
    assert_eq!(stats.live_bytes, 0);

    // Logged-only (D-05): emitting the stats must not panic and is the Phase-1
    // surfacing mechanism (no hard reuse-rate assertion here).
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let h = pool.acquire(64);
    pool.release(h, 64);
    pool.log_stats(); // log::info! — inspected, not asserted as a gate
}

/// `DeviceArray::from_host` → `to_host` returns the original slice on cpu, the
/// allocation routes through the pool (stats reflect the acquire), and `len()`
/// reports the element count.
#[test]
fn device_array_round_trips_host_device_through_pool() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let host: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();

    let allocations_before = pool.stats().allocations;
    let arr: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &host);

    assert_eq!(arr.len(), host.len(), "len reflects element count");
    assert!(!arr.is_empty(), "non-empty input is not empty");
    assert_eq!(
        pool.stats().allocations,
        allocations_before + 1,
        "from_host routes allocation through the pool"
    );

    let got: Vec<f32> = arr.to_host(&pool);
    assert_eq!(got, host, "host->device->host round-trip is lossless on cpu");
}

/// An empty DeviceArray reports len 0 / is_empty.
#[test]
fn device_array_empty_is_empty() {
    let _ = env_logger::builder().is_test(true).try_init();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let arr: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[]);
    assert_eq!(arr.len(), 0);
    assert!(arr.is_empty());
    let got: Vec<f32> = arr.to_host(&pool);
    assert!(got.is_empty());
}
