//! `DeviceArray<R, F>` (FOUND-05) — a typed wrapper over a CubeCL device buffer
//! that carries its element count + dtype and routes allocation through the
//! [`BufferPool`], with host read-back.
//!
//! ## Design (RESEARCH §DeviceArray + Buffer Pool / PATTERNS "No Analog Found")
//! The wrapper type is new mlrs design — CubeCL exposes raw
//! `client.create`/`empty`/`read_one` primitives but no length-carrying typed
//! handle. [`DeviceArray`] wraps a [`cubecl::server::Handle`] plus `len`
//! (element count) and an `F` dtype marker, so read-back size is derived from
//! the carried length and never from caller-supplied geometry (mitigates
//! T-04-01: a wrong length would otherwise read out of bounds).
//!
//! ## Allocation is metered through the pool
//! [`DeviceArray::from_host`] reserves the byte size through
//! [`BufferPool::acquire`] (so the pool's allocation/reuse counters and
//! live/peak bytes account for this array — FOUND-05 / D-04), then uploads the
//! host bytes via the A3-resolved `cubecl::bytes::Bytes` constructor +
//! `client.create`. cubecl 0.10 has no in-place host-write API for an `empty`
//! handle, so the metering handle is returned to the pool's free-list for reuse
//! and the populated `create` handle is the one the array holds. This keeps
//! every device array's footprint visible in [`PoolStats`] while still
//! performing exactly one upload copy (the honest A3 semantics).
//!
//! ## Host read-back (A6)
//! [`DeviceArray::to_host`] reads the buffer back via `client.read_one(handle)`
//! → `bytemuck::cast_slice` into a `Vec<F>` of length `len`. Proven to
//! round-trip on cpu in `crates/mlrs-backend/tests/pool_test.rs`.
//!
//! Tests live in `tests/`, never an in-source `#[cfg(test)]` module (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::server::Handle;

use crate::pool::BufferPool;

/// A typed, length-carrying view of a CubeCL device buffer (FOUND-05).
///
/// `F` is the element type (`f32` / `f64`); the buffer holds `len` contiguous
/// `F` values. The carried `len` is the single source of truth for read-back
/// size (T-04-01 mitigation).
pub struct DeviceArray<R: cubecl::Runtime, F> {
    handle: Handle,
    len: usize,
    _runtime: PhantomData<R>,
    _dtype: PhantomData<F>,
}

impl<R: cubecl::Runtime, F: Pod> DeviceArray<R, F> {
    /// Upload a host slice to a device buffer, routing the allocation through
    /// `pool` and recording `len` + dtype.
    ///
    /// The byte footprint is reserved via [`BufferPool::acquire`] (metering the
    /// allocation / reuse and live/peak bytes), then the host data is uploaded
    /// with a single copy via `cubecl::bytes::Bytes::from_bytes_vec` +
    /// `client.create` (A3 honest single-upload). The metering handle is
    /// released back to the pool for later reuse.
    pub fn from_host(pool: &mut BufferPool<R>, host: &[F]) -> Self {
        let len = host.len();
        let byte_size = size_of_val(host);

        // Meter the allocation through the pool (counters + live/peak bytes).
        // cubecl 0.10 has no in-place write into an `empty` handle, so this
        // handle is returned to the free-list and the populated handle below is
        // the one the array keeps.
        let metering_handle = pool.acquire(byte_size);
        pool.release(metering_handle, byte_size);

        // Single host copy into an owned byte Vec, then hand ownership to
        // CubeCL (A3 — no borrow/no-copy Bytes constructor exists in 0.10).
        let byte_vec: Vec<u8> = bytemuck::cast_slice::<F, u8>(host).to_vec();
        let handle = pool
            .client()
            .create(cubecl::bytes::Bytes::from_bytes_vec(byte_vec));

        Self {
            handle,
            len,
            _runtime: PhantomData,
            _dtype: PhantomData,
        }
    }

    /// Wrap an already-populated CubeCL handle as a `DeviceArray` of `len`
    /// elements, without uploading or metering.
    ///
    /// Used by device-resident producers (e.g. the GEMM host API) that obtain
    /// an output handle from the pool, launch a kernel that writes it, then
    /// hand the result back as a typed length-carrying array (the result stays
    /// on the device — D-05). `len` is the single source of truth for read-back
    /// size (T-04-01), so callers MUST pass the true element count.
    pub fn from_raw(handle: Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _runtime: PhantomData,
            _dtype: PhantomData,
        }
    }

    /// Read the buffer back to a host `Vec<F>` of length [`len`](Self::len).
    ///
    /// Reads via `client.read_one` then reinterprets the bytes with
    /// `bytemuck::cast_slice`. The result length is derived from the carried
    /// `len`, never from caller input (T-04-01). Borrows the same `pool` whose
    /// client owns the buffer.
    pub fn to_host(&self, pool: &BufferPool<R>) -> Vec<F> {
        if self.len == 0 {
            return Vec::new();
        }
        // Handles are cheap ref-counted clones; clone so `self` keeps ownership
        // (read_one consumes the handle).
        let bytes = pool
            .client()
            .read_one(self.handle.clone())
            .expect("device read-back of DeviceArray handle");
        // `cast_slice` is size-checked; take exactly `len` elements to guard
        // against any trailing padding the runtime may have added.
        let view: &[F] = bytemuck::cast_slice(&bytes);
        view[..self.len].to_vec()
    }

    /// Read the buffer back to a host `Vec<F>` while metering the read-back
    /// through the pool's `read_backs` counter (D-10 memory gate).
    ///
    /// Identical result to [`to_host`](Self::to_host), but takes `&mut
    /// BufferPool` and calls [`BufferPool::record_read_back`] first, so each
    /// terminal read-back is a real runtime quantity the Plan-02 memory gate can
    /// assert on (not a code-review claim — RESEARCH §D-10 assertion 2). Prefer
    /// this at terminal reads; [`to_host`](Self::to_host) stays available for
    /// the existing immutable call sites.
    pub fn to_host_metered(&self, pool: &mut BufferPool<R>) -> Vec<F> {
        pool.record_read_back();
        self.to_host(pool)
    }

    /// Number of `F` elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the underlying CubeCL handle (for kernel launches in later
    /// phases).
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}
