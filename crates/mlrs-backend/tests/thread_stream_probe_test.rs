//! Rocm multi-thread panic — diagnostic probes and the answer (2026-08-11).
//!
//! ## Symptom
//!
//! Fanning `fit` out over threads (joblib's `threading` backend, from
//! `mlrs.StackingRegressor(n_jobs=...)`) tore the CubeCL HIP runtime down on
//! rocm gfx1151. First panic, on the shared `DSD-0-0` server thread:
//!
//! ```text
//! cubecl-hip-0.10.0/src/compute/server.rs:84   HipServer::initialize_memory
//!     command.reserve(size).unwrap()  ->  "Unknown error happened during execution"
//! ```
//!
//! reached from `do_create` — an UPLOAD that could not reserve device memory.
//! Everything after it ("Memory page N doesn't exist" at `stream.rs:127`,
//! `create_with_data` at `server.rs:388`, `CallError` at
//! `cubecl-runtime/src/client.rs:105`) is the cascade once that thread is
//! already unwinding, so reading the log top-down misdirects: the page-lookup
//! failures are a CONSEQUENCE, not the cause.
//!
//! ## Root cause (confirmed, not inferred)
//!
//! CubeCL allocates **one GPU stream per OS thread** — `StreamId::current()`
//! reads a thread-local drawn from a global counter
//! (`cubecl-common/src/stream_id.rs`), and `StreamPool` indexes streams as
//! `stream_id.value % max_streams`, where `max_streams` **defaults to 128**
//! (`cubecl-runtime/src/config/streaming.rs`). Each `Stream` owns an
//! INDEPENDENT `MemoryManagement<GpuStorage>` (`cubecl-hip/src/compute/stream.rs`),
//! so a device page belongs to exactly one stream's arena, and streams are
//! never torn down.
//!
//! joblib's `ThreadingBackend` does not reuse threads across `Parallel` calls
//! (verified: cumulative distinct thread ids grow by one per call), and a
//! stacking fit issues one `Parallel` for the members plus one per member
//! inside `cross_val_predict`. A dozen fits therefore mint well over a hundred
//! distinct threads → that many distinct streams → that many independent device
//! arenas resident at once. On gfx1151 the VRAM carve-out is 512 MB and is
//! already ~96% used at idle, so the arenas exhaust the device heap and
//! `reserve()` fails.
//!
//! mlrs's own `BufferPool` is NOT the culprit and is working perfectly: 300
//! single-threaded iterations of an 11.5 MB round-trip cost exactly **1
//! allocation and 599 reuses** (`probe_j`). But its free-list is keyed by byte
//! size alone, so a handle it hands to a new thread still belongs to the OLD
//! stream — the new stream must allocate its own pages regardless, and every
//! launch then pays cross-stream flush/wait alignment.
//!
//! ## The experiment that settles it
//!
//! A `cubecl.toml` next to the working directory capping the stream count:
//!
//! ```toml
//! [streaming]
//! max_streams = 1
//! ```
//!
//! | `max_streams` | outcome of the original crashing script |
//! |---|---|
//! | default (128) | **aborts**, deterministically, at the same fit (3/3 runs) |
//! | 16 / 8 / 4 / 2 / 1 | survives all 12 parallel fits, every result bit-identical to serial |
//!
//! Capping it also made the parallel fits much FASTER (cv=5, `n_jobs=4`:
//! ~3.2 s → ~0.24 s), because the cross-stream alignment goes away. That is the
//! candidate fix: mlrs serializes all device work behind one process-global
//! `Mutex<BufferPool>` (`crates/mlrs-py/src/lib.rs`) anyway, so extra streams
//! buy it nothing and cost it a great deal.
//!
//! Until such a cap ships, `mlrs.StackingRegressor` reduces `n_jobs` to 1 with a
//! `UserWarning` when any composed estimator holds a device handle
//! (`_effective_n_jobs` in `crates/mlrs-py/python/mlrs/ensemble.py`).
//!
//! ## The probes
//!
//! A-E (sequential) and F-H (concurrent) all pass: handle PROVENANCE is handled
//! correctly by cubecl — a buffer allocated on one thread and reused, read, or
//! dropped on another is fine. They are kept as the negative results that
//! narrowed the search. I/J are the pair that isolates the real variable:
//! identical work, 300 times, differing only in whether each iteration runs on a
//! fresh thread.
//!
//! Run them ONE AT A TIME — the first panic kills the shared server thread, so a
//! second probe in the same process would only observe the wreckage:
//!
//! ```bash
//! export ROCM_PATH=/home/user/rocm/opt/rocm HIP_PATH=$ROCM_PATH
//! export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
//! cargo test -p mlrs-backend --release --features rocm \
//!   --test thread_stream_probe_test probe_i_thread_accumulation_big_buffers \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `MLRS_PROBE_ITERS` overrides the I/J iteration count (default 300).
//!
//! `#[ignore]` because these are hardware-specific diagnostics, not a
//! correctness gate: on a single-stream backend (cpu) they are all trivially
//! green and prove nothing.

use std::sync::{Arc, Mutex};

use cubecl::prelude::*;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_kernels::saxpy_kernel;

const N: usize = 4096;

fn launch_dims(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = (n as u32).div_ceil(block);
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim {
            x: block,
            y: 1,
            z: 1,
        },
    )
}

/// `y = 2*x + y` on the device, then read `y` back.
///
/// Deliberately routes every buffer through `pool.acquire`/`release_into` so it
/// exercises the SAME free-list path `mlrs-py`'s estimators do.
fn device_roundtrip(pool: &mut BufferPool<ActiveRuntime>, tag: &str) -> Vec<f32> {
    let x_host: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let y_host: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();

    let x = DeviceArray::<ActiveRuntime, f32>::from_host(pool, &x_host);
    let y = DeviceArray::<ActiveRuntime, f32>::from_host(pool, &y_host);

    let (count, dim) = launch_dims(N);
    saxpy_kernel::launch::<f32, ActiveRuntime>(
        pool.client(),
        count,
        dim,
        2.0f32,
        // SAFETY: length is the validated host slice length; the kernel
        // bounds-checks `tid < x.len()`.
        unsafe { ArrayArg::from_raw_parts(x.handle().clone(), N) },
        unsafe { ArrayArg::from_raw_parts(y.handle().clone(), N) },
    );

    let out = y.to_host(pool);
    println!("[{tag}] out[0..4] = {:?}", &out[..4]);

    // Back to the free-list, so the NEXT acquire of this size reuses them.
    x.release_into(pool);
    y.release_into(pool);
    out
}

fn expected() -> Vec<f32> {
    (0..N)
        .map(|i| 2.0 * (i % 7) as f32 + (i % 5) as f32)
        .collect()
}

// --------------------------------------------------------------------------- //
// (A) free-list hand-over: one shared pool, two threads, NO concurrency
// --------------------------------------------------------------------------- //

/// Thread A fills the free-list; thread B drains it. Never concurrent.
///
/// This is EXACTLY `mlrs-py`'s shape: `lock_pool()` is taken inside
/// `py.detach` and held for the whole `fit`, so two Python worker threads are
/// fully serialized on the mutex — they never launch kernels at the same time.
/// The only thing that crosses the thread boundary is a `Handle` whose page
/// lives in the other thread's stream.
///
/// If THIS panics, concurrency is a red herring and the shared free-list is the
/// bug.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_a_shared_pool_handover() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));
    let want = expected();

    let a = {
        let pool = pool.clone();
        std::thread::spawn(move || device_roundtrip(&mut pool.lock().unwrap(), "A"))
    };
    let got_a = a.join().expect("thread A panicked");
    assert_eq!(got_a, want, "thread A result");

    // Thread B now acquires the very handles thread A released.
    let b = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let mut guard = pool.lock().unwrap();
            let before = guard.stats();
            let got = device_roundtrip(&mut guard, "B");
            let after = guard.stats();
            println!(
                "[B] reuses {} -> {} (a hand-over happened iff this rose)",
                before.reuses, after.reuses
            );
            assert!(
                after.reuses > before.reuses,
                "probe is vacuous: thread B allocated fresh buffers instead of \
                 reusing thread A's, so no handle crossed the thread boundary"
            );
            got
        })
    };
    let got_b = b.join().expect("thread B panicked");
    assert_eq!(got_b, want, "thread B result");
}

// --------------------------------------------------------------------------- //
// (B) control for (A): two threads, PRIVATE pools — no handle ever crosses
// --------------------------------------------------------------------------- //

/// Same two threads and the same two streams, but nothing is shared.
///
/// Green here + red in `probe_a` isolates the free-list hand-over. Red here too
/// means the mere existence of a second stream is enough, and the pool is
/// innocent.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_b_private_pools_two_threads() {
    let want = expected();
    for tag in ["A", "B"] {
        let want = want.clone();
        let h = std::thread::spawn(move || {
            let mut pool = BufferPool::new(runtime::active_client());
            let got = device_roundtrip(&mut pool, tag);
            assert_eq!(got, want, "thread {tag} result");
        });
        h.join().unwrap_or_else(|_| panic!("thread {tag} panicked"));
    }
}

// --------------------------------------------------------------------------- //
// (C) cross-thread drop: allocate on a worker, drop on the main thread
// --------------------------------------------------------------------------- //

/// A `DeviceArray` outlives the thread that created it.
///
/// This is what a fitted mlrs estimator does: its device buffers are allocated
/// inside a worker thread's `fit` and freed later, when Python drops the
/// estimator on whatever thread happens to run the GC. If the free is routed to
/// the dropping thread's stream, the page is not in that stream's table.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_c_cross_thread_drop() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));

    // Worker allocates and returns the array WITHOUT releasing it.
    let escaped = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let mut guard = pool.lock().unwrap();
            let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
            DeviceArray::<ActiveRuntime, f32>::from_host(&mut guard, &host)
        })
        .join()
        .expect("allocating thread panicked")
    };

    // Main thread reads it (cross-stream READ) ...
    {
        let guard = pool.lock().unwrap();
        let got = escaped.to_host(&guard);
        assert_eq!(got[7], 7.0, "cross-thread read-back");
    }
    // ... and then drops it (cross-stream FREE).
    drop(escaped);

    // Prove the runtime still works afterwards.
    let mut guard = pool.lock().unwrap();
    assert_eq!(device_roundtrip(&mut guard, "post-drop"), expected());
}

// --------------------------------------------------------------------------- //
// (D) stream-count pressure: many threads, private pools
// --------------------------------------------------------------------------- //

/// 64 sequential threads, each with its own pool and therefore its own stream.
///
/// `max_streams` defaults to 128 and each stream owns an independent
/// `MemoryManagement`, so on an APU this is 64 separate device-memory arenas.
/// Red here with (A)/(B) green would mean the failure is resource exhaustion,
/// not handle provenance.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_d_many_threads_private_pools() {
    let want = expected();
    for i in 0..64 {
        let want = want.clone();
        std::thread::spawn(move || {
            let mut pool = BufferPool::new(runtime::active_client());
            assert_eq!(device_roundtrip(&mut pool, &format!("t{i}")), want);
        })
        .join()
        .unwrap_or_else(|_| panic!("thread {i} panicked"));
    }
}

// --------------------------------------------------------------------------- //
// (E) the real shape: many threads over ONE shared pool
// --------------------------------------------------------------------------- //

/// 64 sequential threads over the shared pool — hand-over on every iteration.
///
/// The scaled-up form of (A); the stacking crash appeared only once the fan-out
/// grew (2 members / cv=5 survived, 3 members / cv=10 did not), so a probe that
/// only runs two threads may under-report.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_e_shared_pool_many_threads() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));
    let want = expected();
    for i in 0..64 {
        let pool = pool.clone();
        let want = want.clone();
        std::thread::spawn(move || {
            let mut guard = pool.lock().unwrap();
            assert_eq!(device_roundtrip(&mut guard, &format!("t{i}")), want);
        })
        .join()
        .unwrap_or_else(|_| panic!("thread {i} panicked"));
    }
    println!("final pool stats: {:?}", pool.lock().unwrap().stats());
}

// --------------------------------------------------------------------------- //
// (F/G/H) TRUE concurrency — probes A-E all join before the next thread starts
// --------------------------------------------------------------------------- //
//
// A-E establish that handle PROVENANCE is handled correctly: cubecl records the
// originating `StreamId` on every `Handle` and resolves pages against that
// stream, so a buffer allocated on one thread and reused (or dropped) on
// another is fine as long as the two never overlap in time. Whatever broke the
// runtime therefore needs threads running AT THE SAME TIME. These three probe
// that, at a buffer size in the range the stacking fit actually used
// (60000 x 48 f32 ~ 11 MB, vs 16 KB above) so memory pressure is comparable.

const BIG: usize = 60_000 * 48;

fn big_roundtrip(pool: &mut BufferPool<ActiveRuntime>, tag: &str) -> f32 {
    let x_host: Vec<f32> = (0..BIG).map(|i| (i % 7) as f32).collect();
    let y_host: Vec<f32> = (0..BIG).map(|i| (i % 5) as f32).collect();
    let x = DeviceArray::<ActiveRuntime, f32>::from_host(pool, &x_host);
    let y = DeviceArray::<ActiveRuntime, f32>::from_host(pool, &y_host);
    let (count, dim) = launch_dims(BIG);
    saxpy_kernel::launch::<f32, ActiveRuntime>(
        pool.client(),
        count,
        dim,
        2.0f32,
        // SAFETY: as in `device_roundtrip`.
        unsafe { ArrayArg::from_raw_parts(x.handle().clone(), BIG) },
        unsafe { ArrayArg::from_raw_parts(y.handle().clone(), BIG) },
    );
    let out = y.to_host(pool);
    let checksum = out[0] + out[1] + out[BIG - 1];
    println!("[{tag}] checksum = {checksum}");
    x.release_into(pool);
    y.release_into(pool);
    checksum
}

/// Threads run device work SIMULTANEOUSLY, each with its own pool and stream.
///
/// No shared mlrs state at all — if this is red, concurrent multi-stream use of
/// the HIP server is itself the defect and no amount of pool discipline helps.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_f_concurrent_private_pools() {
    let handles: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                let mut pool = BufferPool::new(runtime::active_client());
                for r in 0..8 {
                    big_roundtrip(&mut pool, &format!("t{i}r{r}"));
                }
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
    }
}

/// The exact `mlrs-py` shape: threads contend on ONE global `Mutex<BufferPool>`.
///
/// The lock serializes the compute, but the free-list still hands stream-A pages
/// to thread B, and the threads' `Handle` drops and allocations interleave at
/// the lock boundary.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_g_concurrent_shared_pool() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let pool = pool.clone();
            std::thread::spawn(move || {
                for r in 0..8 {
                    let mut guard = pool.lock().unwrap();
                    big_roundtrip(&mut guard, &format!("t{i}r{r}"));
                }
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
    }
    println!("final pool stats: {:?}", pool.lock().unwrap().stats());
}

/// A thread DROPS foreign-stream buffers while another thread is mid-kernel.
///
/// This is the one interleaving `mlrs-py`'s global mutex does NOT cover: a
/// `Handle`'s `Drop` runs wherever Python's GC happens to collect the estimator,
/// with no pool lock held, concurrently with another thread's compute.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_h_concurrent_foreign_drop() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));

    // Allocate a batch of arrays on a dedicated thread (its own stream) and let
    // them escape, unreleased.
    let escaped: Vec<DeviceArray<ActiveRuntime, f32>> = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let mut guard = pool.lock().unwrap();
            let host: Vec<f32> = (0..BIG).map(|i| (i % 11) as f32).collect();
            (0..8)
                .map(|_| DeviceArray::<ActiveRuntime, f32>::from_host(&mut guard, &host))
                .collect()
        })
        .join()
        .expect("allocating thread panicked")
    };

    let worker = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            for r in 0..16 {
                let mut guard = pool.lock().unwrap();
                big_roundtrip(&mut guard, &format!("worker r{r}"));
            }
        })
    };

    // Meanwhile, on THIS thread (a third stream), free the foreign pages one by
    // one without ever taking the pool lock.
    for (i, arr) in escaped.into_iter().enumerate() {
        drop(arr);
        println!("[dropper] freed foreign array {i}");
        std::thread::yield_now();
    }

    worker.join().expect("worker thread panicked");
    let mut guard = pool.lock().unwrap();
    big_roundtrip(&mut guard, "post-drop");
}

// --------------------------------------------------------------------------- //
// (I/J) stream ACCUMULATION — the hypothesis the Python repro actually points at
// --------------------------------------------------------------------------- //
//
// Re-running the original crash with `RUST_BACKTRACE=1` showed the FIRST panic
// is not the page lookup at all — it is
//
//   HipServer::initialize_memory -> command.reserve(size).unwrap()
//     "Unknown error happened during execution"     (server.rs:84)
//
// reached from `do_create`, i.e. an UPLOAD failing to reserve device memory.
// Every "Memory page N doesn't exist" that follows is the cascade after the
// server thread is already unwinding.
//
// That points at exhaustion, and there is a mechanism for it: joblib's
// `ThreadingBackend` builds a FRESH thread pool per `Parallel` call, every new
// OS thread draws a new `StreamId`, and every `StreamId` (mod 128) materializes
// a `Stream` that owns an INDEPENDENT `MemoryManagement<GpuStorage>`. Streams
// are never torn down, and a page belongs to exactly one stream's arena, so N
// threads over a session means up to N copies of the working set resident at
// once. On gfx1151 — an APU whose device memory is carved out of system RAM —
// that is a hard ceiling.
//
// `probe_i` scales the thread COUNT (each short-lived, as joblib's are) at a
// realistic buffer size; `probe_j` is the identical amount of work on ONE
// thread. Green J + red I is the exhaustion mechanism, isolated.

/// Iterations for the accumulation probes: comfortably past `max_streams` (128)
/// so wrap-around is exercised too.
fn accum_iters() -> usize {
    std::env::var("MLRS_PROBE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

/// `ACCUM_ITERS` short-lived threads, one after another, sharing one pool.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_i_thread_accumulation_big_buffers() {
    let pool = Arc::new(Mutex::new(BufferPool::new(runtime::active_client())));
    for i in 0..accum_iters() {
        let t0 = std::time::Instant::now();
        let thread_pool = pool.clone();
        std::thread::spawn(move || {
            let mut guard = thread_pool.lock().unwrap();
            big_roundtrip(&mut guard, &format!("thread{i}"));
        })
        .join()
        .unwrap_or_else(|_| panic!("died after {i} distinct threads/streams"));
        println!(
            "thread {i}: {:.1} ms  (cumulative distinct streams = {})",
            t0.elapsed().as_secs_f64() * 1e3,
            i + 1
        );
    }
    println!("final pool stats: {:?}", pool.lock().unwrap().stats());
}

/// The control: the same `ACCUM_ITERS` roundtrips, all on the calling thread.
///
/// Same total device work, same buffer sizes, ONE stream. If this is green and
/// `probe_i` is red, the thread count is the variable that matters — not the
/// workload.
#[test]
#[ignore = "rocm hardware diagnostic; see module docs"]
fn probe_j_same_work_single_thread() {
    let mut pool = BufferPool::new(runtime::active_client());
    for i in 0..accum_iters() {
        big_roundtrip(&mut pool, &format!("iter{i}"));
        if i % 25 == 0 {
            println!("--- survived {i} iterations; {:?}", pool.stats());
        }
    }
    println!("final pool stats: {:?}", pool.stats());
}
