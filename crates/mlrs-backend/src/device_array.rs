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
        let mut bytes = cubecl::bytes::Bytes::from_bytes_vec(byte_vec);

        // `MLRS_UPLOAD_PINNED=1` moves the staging buffer to PAGE-LOCKED host
        // memory before the transfer, which lets the driver DMA straight out of
        // it instead of bouncing through its own internal pinned buffer.
        //
        // Off by default, and MEASURED to belong that way on cuda. It buys a
        // faster transfer with an extra full host copy, so it only pays when
        // the transfer dominates that copy — and on a Colab T4 it does not:
        //
        // | 102.4 MB          | ms    | GB/s |
        // |-------------------|-------|------|
        // | transfer alone    | 266.1 | 0.38 |
        // | pinned + transfer | 197.7 | 0.51 |
        //
        // The transfer half genuinely improves 1.35×, but the extra host copy
        // that buys it costs ~72 ms (host memcpy on that VM runs at 1.39 GB/s),
        // which more than eats the gain. End to end the whole `d=256` fit went
        // 316.1 ms → 336.3 ms with this on — a REGRESSION, exactly the way the
        // isolated column said it would not. Trust the end-to-end number.
        //
        // `cubecl-cuda`'s `reserve_cpu` also refuses pinned memory above 100 MB
        // unless explicitly marked, so the 102.4 MB rung may not even be pinned.
        if crate::abflag::is_on("MLRS_UPLOAD_PINNED") {
            pool.client().staging(std::iter::once(&mut bytes), false);
        }
        let handle = pool.client().create(bytes);

        Self {
            handle,
            len,
            _runtime: PhantomData,
            _dtype: PhantomData,
        }
    }

    /// Upload several host slices CONCATENATED into one device buffer, with the
    /// same single host copy [`from_host`](Self::from_host) performs.
    ///
    /// For a caller holding `k` separate columns that a kernel wants to read as
    /// one `k × n` buffer, the obvious spelling — build a packed `Vec<F>`, then
    /// `from_host` it — copies the data TWICE: once to pack and once inside
    /// `from_host`. This packs directly into the byte buffer that becomes
    /// `Bytes`, so the total is one copy of `Σ chunk.len()`, exactly what `k`
    /// separate `from_host` calls would have cost between them.
    ///
    /// That is not a micro-optimisation. Measured on rocm (gfx1151) for
    /// `VotingRegressor`'s `predict` at n = 10⁶, k = 3, the double copy cost
    /// **1.8 ms** — enough to turn a launch-count win into a net regression
    /// (8.08 ms chained → 9.85 ms double-copied → see `docs/voting.md` for where
    /// the single-copy version lands).
    ///
    /// Returns an array of `Σ chunk.len()` elements, chunk `j` occupying
    /// `[Σ_{i<j} len_i, Σ_{i<=j} len_i)`.
    pub fn from_host_chunks(pool: &mut BufferPool<R>, chunks: &[&[F]]) -> Self {
        let len: usize = chunks.iter().map(|c| c.len()).sum();
        let byte_size = len * size_of::<F>();

        // Metered exactly as `from_host` does, for the same reason.
        let metering_handle = pool.acquire(byte_size);
        pool.release(metering_handle, byte_size);

        let mut byte_vec: Vec<u8> = Vec::with_capacity(byte_size);
        for chunk in chunks {
            byte_vec.extend_from_slice(bytemuck::cast_slice::<F, u8>(chunk));
        }
        let mut bytes = cubecl::bytes::Bytes::from_bytes_vec(byte_vec);

        // Same opt-in pinned-staging knob as `from_host`; see its comment for
        // why it is off by default.
        if crate::abflag::is_on("MLRS_UPLOAD_PINNED") {
            pool.client().staging(std::iter::once(&mut bytes), false);
        }
        let handle = pool.client().create(bytes);

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

    /// The buffer's TRUE byte footprint (`len * size_of::<F>()`).
    ///
    /// This is the single source of truth for the size a handle was acquired at,
    /// so it is the ONLY size a release should ever file the handle under (WR-07):
    /// releasing under a guessed/wrong size pollutes the free-list and causes a
    /// later same-keyed `acquire` to hand back a buffer of a different real size
    /// (over/under-read). Mirrors the `len`-as-source-of-truth invariant
    /// [`from_host`](Self::from_host) / [`from_raw`](Self::from_raw) maintain for
    /// read-back.
    pub fn byte_size(&self) -> usize {
        self.len * size_of::<F>()
    }

    /// Return this array's buffer to the pool's free-list for later reuse,
    /// consuming the array (CR-02 / WR-07).
    ///
    /// The handle is filed under the array's OWN [`byte_size`](Self::byte_size)
    /// — the true acquisition size — so the free-list key is always correct
    /// (WR-07: no guessed size). Taking `self` by value enforces at the type
    /// level that the array cannot be read after its buffer is released: a
    /// released buffer may be handed to a later `acquire`, so reusing the array
    /// afterwards would alias live memory. Call this ONLY on genuinely-transient
    /// scratch whose consuming kernel has already been launched — NEVER on a
    /// buffer that is returned to the caller or otherwise outlives the call.
    pub fn release_into(self, pool: &mut BufferPool<R>) {
        let bytes = self.byte_size();
        pool.release(self.handle, bytes);
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
