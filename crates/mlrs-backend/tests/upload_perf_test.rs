//! Host→device upload wall-clock probe (RIDGE-POS-CUDA).
//!
//! ## Why this exists
//! The T4 whole-fit ladder showed the design UPLOAD is 85–91% of every
//! device-arm `Ridge(positive=True)` fit, at **0.33–0.92 GB/s** — ten to twenty
//! times below what a T4's PCIe link does. And the throughput FALLS as the
//! operand grows (0.92 GB/s at 2.6 MB → 0.33 GB/s at 102 MB), which is not the
//! signature of a slow link (that would be a constant rate) but of a per-call
//! allocation cost growing with size.
//!
//! `DeviceArray::from_host` is four distinct costs stacked, and the ladder
//! cannot tell them apart:
//!
//! 1. the `to_vec()` host copy (allocate `n·d` bytes, fault them in, memcpy),
//! 2. the pool's metering `acquire`/`release` (a real `client.empty` of the
//!    same size, whose buffer is then discarded),
//! 3. the device allocation inside `client.create`,
//! 4. the actual PCIe transfer.
//!
//! This probe times each in isolation so the fix targets the one that is
//! actually expensive rather than the one that looks suspicious.
//!
//! ```text
//! cargo test -p mlrs-backend --release --features cuda \
//!   --test upload_perf_test -- --ignored --nocapture
//! ```
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use cubecl::bytes::Bytes;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Megabytes per second for `bytes` moved in `secs`.
fn gbs(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / secs
}

/// Force the stream to drain WITHOUT reading back the large operand.
///
/// `client.sync()` returns a future and does nothing when dropped, and reading
/// the uploaded buffer itself would add a device→host transfer of the very size
/// under test. Reading a 4-byte buffer on the same stream orders after the
/// pending upload and costs nothing measurable.
fn drain(pool: &mut BufferPool<ActiveRuntime>, probe: &DeviceArray<ActiveRuntime, f32>) {
    let v = probe.to_host(pool);
    assert!(v[0].is_finite());
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn upload_cost_breakdown() {
    let reps: usize = std::env::var("MLRS_POS_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client.clone());
    let probe: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[1.0f32]);

    println!(
        "backend={} reps={reps}  (min-of-N, ms; GB/s in parentheses)",
        mlrs_backend::capability::active_backend_name()
    );
    println!(
        "{:>8} | {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "MB",
        "1 host to_vec",
        "2 empty(alloc)",
        "3 create",
        "4 staged create",
        "5 from_host",
        "6 chunked 32MB"
    );

    // The ladder's actual operand sizes: 100k×16, 10k×256/100k×64, 500k×16,
    // 100k×256 — the last is deliberately just over cubecl's 100 MB pinned-
    // memory cutoff (`reserve_cpu`), which is a candidate explanation for the
    // throughput collapse at exactly that rung.
    for mb in [6.4f64, 25.6, 32.0, 102.4] {
        let elems = (mb * 1024.0 * 1024.0 / 4.0) as usize;
        let bytes = elems * 4;
        let src: Vec<f32> = (0..elems).map(|i| (i % 1000) as f32 * 0.001).collect();

        let mut t_tovec = f64::INFINITY;
        let mut t_empty = f64::INFINITY;
        let mut t_create = f64::INFINITY;
        let mut t_staged = f64::INFINITY;
        let mut t_full = f64::INFINITY;
        let mut t_chunk = f64::INFINITY;

        for _ in 0..reps {
            // 1. Host copy alone — allocation + first-touch faults + memcpy.
            let t0 = Instant::now();
            let v: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&src).to_vec();
            t_tovec = t_tovec.min(t0.elapsed().as_secs_f64());
            std::hint::black_box(&v);

            // 2. Device allocation alone, no transfer.
            let t0 = Instant::now();
            let h = client.empty(bytes);
            drain(&mut pool, &probe);
            t_empty = t_empty.min(t0.elapsed().as_secs_f64());
            drop(h);

            // 3. create() from an ALREADY-BUILT byte vec: device alloc + PCIe,
            //    with the host copy of step 1 excluded.
            let t0 = Instant::now();
            let h = client.create(Bytes::from_bytes_vec(v));
            drain(&mut pool, &probe);
            t_create = t_create.min(t0.elapsed().as_secs_f64());
            drop(h);

            // 4. Same, but moved to PINNED host memory first. `reserve_cpu`
            //    refuses pinned above 100 MB unless explicitly marked, so this
            //    column is exactly where that cutoff would show itself.
            let v2: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&src).to_vec();
            let mut b2 = Bytes::from_bytes_vec(v2);
            let t0 = Instant::now();
            client.staging(std::iter::once(&mut b2), false);
            let h = client.create(b2);
            drain(&mut pool, &probe);
            t_staged = t_staged.min(t0.elapsed().as_secs_f64());
            drop(h);

            // 5. The shipped path, end to end.
            let t0 = Instant::now();
            let dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &src);
            drain(&mut pool, &probe);
            t_full = t_full.min(t0.elapsed().as_secs_f64());
            dev.release_into(&mut pool);

            // 6. The same bytes as SEPARATE 32 MB row chunks. The Gram and the
            //    column means are both row-wise reductions, so a chunked design
            //    needs no contiguous device buffer at all — each chunk can be
            //    uploaded and reduced on its own. If the per-call cost is what
            //    scales badly (allocating and faulting in one huge host buffer,
            //    and landing above cubecl's 100 MB pinned cutoff), then several
            //    small uploads beat one large one and this column shows it.
            let t0 = Instant::now();
            let rows_per_chunk = (32 * 1024 * 1024 / 4).min(elems).max(1);
            let mut chunks: Vec<DeviceArray<ActiveRuntime, f32>> = Vec::new();
            let mut off = 0usize;
            while off < elems {
                let end = (off + rows_per_chunk).min(elems);
                chunks.push(DeviceArray::from_host(&mut pool, &src[off..end]));
                off = end;
            }
            drain(&mut pool, &probe);
            t_chunk = t_chunk.min(t0.elapsed().as_secs_f64());
            for c in chunks {
                c.release_into(&mut pool);
            }
        }

        let cell = |s: f64| format!("{:.1} ({:.2})", s * 1e3, gbs(bytes, s));
        println!(
            "{mb:>8.1} | {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
            cell(t_tovec),
            cell(t_empty),
            cell(t_create),
            cell(t_staged),
            cell(t_full),
            cell(t_chunk),
        );
    }
}
