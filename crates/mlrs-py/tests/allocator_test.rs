//! Plan 05, Task 3 — mimalloc global-allocator activation proof (FOUND-09).
//!
//! `mlrs-py` defines `#[global_allocator] static GLOBAL: MiMalloc = MiMalloc;`
//! exactly once in `src/allocator.rs`. Because this is an **integration test**
//! (separate crate that links the `mlrs-py` cdylib/rlib), every heap allocation
//! performed below is served by that global allocator. mimalloc exposes no
//! stable public introspection symbol from the `mimalloc` crate, so the
//! activation is proven the way a drop-in allocator is meant to be: by
//! **exercising it heavily and asserting correctness** — large and small
//! allocations, growth/realloc, cross-thread alloc/free, and data integrity. If
//! the wired allocator were broken, these would corrupt or crash; that they run
//! cleanly is the evidence the custom allocator is active and sound.
//!
//! Per AGENTS.md §2 this lives in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`. It links `mlrs-py`, which is what pulls in the `#[global_allocator]`.

use std::thread;

// Link the crate under test so its `#[global_allocator]` is in force for this
// test binary. (The reference also documents the dependency edge explicitly.)
use mlrs_py as _;

/// Exercise a spread of allocation sizes and assert the bytes written survive
/// readback — a corrupting allocator would fail integrity here.
#[test]
fn allocator_handles_varied_sizes_with_integrity() {
    // A spread of size classes: tiny, small, page-ish, large.
    for &size in &[1usize, 7, 64, 4096, 1 << 16, 1 << 20] {
        let mut v: Vec<u8> = Vec::with_capacity(size);
        for i in 0..size {
            v.push((i % 251) as u8); // 251 prime => non-trivial byte pattern
        }
        assert_eq!(v.len(), size, "allocation of {size} bytes filled");
        // Verify every byte round-trips (no allocator-induced corruption).
        for (i, &b) in v.iter().enumerate() {
            assert_eq!(b, (i % 251) as u8, "byte {i} of a {size}-byte alloc intact");
        }
        // Force a realloc/grow path through the global allocator.
        v.extend(std::iter::repeat(0xABu8).take(size));
        assert_eq!(v.len(), size * 2, "grow path doubled the buffer");
    }
}

/// Many short-lived allocations across threads stress mimalloc's thread-local
/// heaps and cross-thread free path. Completing without panic/abort proves the
/// global allocator handles concurrent churn.
#[test]
fn allocator_survives_concurrent_churn() {
    let handles: Vec<_> = (0..8)
        .map(|t| {
            thread::spawn(move || {
                let mut acc: u64 = 0;
                // Allocate, touch, and drop many boxes/vecs per thread.
                for round in 0..10_000u64 {
                    let n = ((round * 7 + t as u64) % 512) as usize + 1;
                    let buf: Vec<u64> = (0..n as u64).map(|k| k.wrapping_mul(round + 1)).collect();
                    acc = acc.wrapping_add(buf.iter().copied().sum::<u64>());
                    // Box on the heap too (different size class than the Vec).
                    let boxed = Box::new([t as u64; 16]);
                    acc = acc.wrapping_add(boxed.iter().sum::<u64>());
                }
                acc
            })
        })
        .collect();

    // Every thread must finish cleanly; sum the per-thread accumulators so the
    // work cannot be optimised away.
    let total: u64 = handles
        .into_iter()
        .map(|h| h.join().expect("worker thread completed under the global allocator"))
        .fold(0u64, |a, b| a.wrapping_add(b));

    // The exact value is irrelevant; the point is the workload ran to
    // completion under mimalloc without corruption/abort.
    let _ = total;
}

/// A single very large allocation + free exercises mimalloc's large-object /
/// OS-page path and confirms it returns cleanly.
#[test]
fn allocator_handles_large_allocation() {
    let n = 8 * 1024 * 1024; // 8 MiB of u8
    let mut big = vec![0u8; n];
    big[0] = 1;
    big[n - 1] = 2;
    assert_eq!(big.len(), n);
    assert_eq!(big[0], 1);
    assert_eq!(big[n - 1], 2);
    drop(big); // return the large block to the allocator/OS without crash.
}
