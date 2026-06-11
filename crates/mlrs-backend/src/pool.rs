//! Buffer-reuse pool (FOUND-05 / D-04) — an mlrs-level free-list over reclaimed
//! CubeCL handles, with a logged-only stats/counters API (D-05).
//!
//! ## Design (RESEARCH Pattern 6)
//! [`BufferPool`] keeps a `HashMap<usize, Vec<Handle>>` keyed by **byte size**:
//! `acquire(size)` pops a reusable handle of exactly that size (a *reuse*) or
//! falls back to `client.empty(size)` (an *allocation*); `release(handle, size)`
//! returns the handle to the free-list. This is a reuse layer **on top of**
//! CubeCL's own allocator — we deliberately do NOT tune CubeCL's
//! `MemoryConfiguration` in Phase 1 (RESEARCH Open Question 4); the simplest
//! correct mlrs-level free-list is sufficient because Phase-1 counters are
//! logged, not asserted.
//!
//! ## Counters are LOGGED ONLY (D-05)
//! [`PoolStats`] tracks `allocations` / `reuses` / `peak_bytes` / `live_bytes`.
//! Phase 1 emits them via [`BufferPool::log_stats`] (and on `Drop`) with
//! `log::info!` — it does NOT assert a reuse-rate threshold. The trivial smoke
//! workloads do not exercise realistic allocation patterns yet, so hard memory
//! assertions are deferred to Phase 2 (D-05).
//!
//! Tests live in `crates/mlrs-backend/tests/pool_test.rs` (never an in-source
//! `#[cfg(test)]` test module — AGENTS.md §2).

use std::collections::HashMap;

use cubecl::server::Handle;

use crate::runtime::Client;

/// Logged-only buffer-pool counters (D-05).
///
/// All four counters are surfaced via `log::info!` at a phase boundary
/// ([`BufferPool::log_stats`]) or on `Drop`; none is used as a hard reuse-rate
/// gate in Phase 1 (the realistic-allocation assertions are a Phase-2 concern).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of fresh device allocations (`client.empty` calls / free-list
    /// misses).
    pub allocations: u64,
    /// Number of acquires served from the free-list (reuses of a released
    /// handle of matching byte size).
    pub reuses: u64,
    /// High-water mark of `live_bytes` observed over the pool's lifetime. Never
    /// decreases.
    pub peak_bytes: u64,
    /// Currently-live bytes: increased on `acquire`, decreased on `release`.
    pub live_bytes: u64,
    /// Number of device→host read-backs performed through a metered read path
    /// (bumped by [`BufferPool::record_read_back`]). Enables the D-10 memory
    /// gate (Plan 02 asserts read-back count, not just logs it).
    pub read_backs: u64,
}

/// An mlrs-level buffer-reuse pool over CubeCL device handles, keyed by byte
/// size (FOUND-05 / D-04).
///
/// Reuse is per exact byte size: a handle released at size `n` is only handed
/// back out by an `acquire(n)`. Distinct sizes never alias. The pool owns the
/// active [`Client`]; allocation misses route through `client.empty`.
pub struct BufferPool<R: cubecl::Runtime> {
    client: cubecl::client::ComputeClient<R>,
    /// `byte_size -> reusable handles` free-list.
    free: HashMap<usize, Vec<Handle>>,
    stats: PoolStats,
}

impl BufferPool<crate::runtime::ActiveRuntime> {
    /// Construct a pool over the active-runtime [`Client`].
    ///
    /// Spelled against the [`Client`] alias so call sites never write the
    /// `ComputeClient<R>` generics (A6). For non-active runtimes use
    /// [`BufferPool::with_client`].
    pub fn new(client: Client) -> Self {
        Self::with_client(client)
    }
}

impl<R: cubecl::Runtime> BufferPool<R> {
    /// Construct a pool over an explicit [`cubecl::client::ComputeClient`].
    pub fn with_client(client: cubecl::client::ComputeClient<R>) -> Self {
        Self {
            client,
            free: HashMap::new(),
            stats: PoolStats::default(),
        }
    }

    /// Borrow the owned compute client (used by [`DeviceArray`] to upload /
    /// read back).
    ///
    /// [`DeviceArray`]: crate::device_array::DeviceArray
    pub fn client(&self) -> &cubecl::client::ComputeClient<R> {
        &self.client
    }

    /// Snapshot the current counters (logged-only — D-05).
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Acquire a device buffer of `size_bytes`.
    ///
    /// Reuses a released handle of the exact same byte size if one is available
    /// (`reuses += 1`); otherwise allocates a fresh buffer via `client.empty`
    /// (`allocations += 1`). Either way `live_bytes` rises by `size_bytes` and
    /// `peak_bytes` tracks the new high-water mark.
    pub fn acquire(&mut self, size_bytes: usize) -> Handle {
        let handle = match self.free.get_mut(&size_bytes).and_then(Vec::pop) {
            Some(reused) => {
                self.stats.reuses += 1;
                reused
            }
            None => {
                self.stats.allocations += 1;
                self.client.empty(size_bytes)
            }
        };
        self.stats.live_bytes += size_bytes as u64;
        if self.stats.live_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.live_bytes;
        }
        handle
    }

    /// Return a handle of `size_bytes` to the free-list for later reuse.
    ///
    /// Decreases `live_bytes` by `size_bytes` (saturating, so a mismatched
    /// release can never underflow). `peak_bytes` is a high-water mark and is
    /// not reduced.
    pub fn release(&mut self, handle: Handle, size_bytes: usize) {
        self.free.entry(size_bytes).or_default().push(handle);
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(size_bytes as u64);
    }

    /// Record a single device→host read-back (`read_backs += 1`).
    ///
    /// Mirrors the `acquire`/`release` counter-bump idiom. Call this at each
    /// terminal read-back so the pool's `read_backs` counter is a real runtime
    /// quantity the D-10 memory gate (Plan 02) can assert on, rather than a
    /// code-review claim. The metered read path
    /// [`DeviceArray::to_host_metered`] routes through here.
    ///
    /// [`DeviceArray::to_host_metered`]: crate::device_array::DeviceArray::to_host_metered
    pub fn record_read_back(&mut self) {
        self.stats.read_backs += 1;
    }

    /// Emit the current counters via `log::info!` (D-05, logged-only).
    ///
    /// This is the Phase-1 surfacing mechanism for pool behaviour. It does NOT
    /// assert a reuse-rate threshold — hard memory assertions are deferred to
    /// Phase 2.
    pub fn log_stats(&self) {
        log::info!("pool stats: {:?}", self.stats);
    }
}

impl<R: cubecl::Runtime> Drop for BufferPool<R> {
    fn drop(&mut self) {
        // Logged-only at the phase/scope boundary (D-05).
        self.log_stats();
    }
}
