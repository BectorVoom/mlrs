//! KNN-01 tile-size sensitivity probe.
//!
//! `neighbor_indices` chunks the query rows so the intermediate distance matrix
//! cannot blow up (a 10_000 × 100_000 f32 matrix is 4 GB and simply fails to
//! allocate). This probe measures what that chunking COSTS: it runs the same
//! total `n_query × n_train` work at several tile heights and reports the
//! per-config wall clock, so the tile budget is chosen from a measurement
//! instead of from a plausible-sounding constant.
//!
//! The suspicion it is built to test: `prims::distance` recomputes the
//! `n_train`-row squared-norm term on EVERY call, so a tiled loop repeats that
//! reduction once per tile. If that term dominates, small tiles are pathological
//! and the budget must stay as large as memory allows.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features wgpu \
//!   --test knn_tile_probe_test -- --ignored --nocapture
//! ```
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance_with_ynorm;
use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::{self, ActiveRuntime};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0)
        .collect()
}

/// Time the full `distance → top_k` pipeline over `n_query` rows, cut into tiles
/// of `tile` rows each.
fn run(n_train: usize, d: usize, n_query: usize, k: usize, tile: usize, hoist: bool) -> f64 {
    let x = uniform(n_train * d, 42);
    let xq = uniform(n_query * d, 7);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

    let val_handle = pool.acquire(n_query * k * size_of::<f32>());
    let idx_handle = pool.acquire(n_query * k * size_of::<u32>());

    let t0 = Instant::now();
    // The term under test: `‖y_j‖²` over the whole training set, either computed
    // ONCE here or left to `distance` to recompute on every tile.
    let ynorm = if hoist {
        Some(
            row_reduce::<f32>(&mut pool, &x_dev, n_train, d, ScalarOp::SumSq, ReducePath::Shared)
                .expect("row_reduce")
                .expect("shared path"),
        )
    } else {
        None
    };
    let mut start = 0usize;
    while start < n_query {
        let rows = tile.min(n_query - start);
        let xq_tile: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(
            xq_dev
                .handle()
                .clone()
                .offset_start((start * d * size_of::<f32>()) as u64),
            rows * d,
        );
        let ov: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(
            val_handle
                .clone()
                .offset_start((start * k * size_of::<f32>()) as u64),
            rows * k,
        );
        let oi: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_raw(
            idx_handle
                .clone()
                .offset_start((start * k * size_of::<u32>()) as u64),
            rows * k,
        );
        let dist = distance_with_ynorm::<f32>(
            &mut pool,
            &xq_tile,
            (rows, d),
            &x_dev,
            (n_train, d),
            false,
            None,
            ynorm.as_ref(),
        )
        .expect("distance");
        let _ = top_k::<f32>(&mut pool, &dist, rows, n_train, k, true, Some(ov), Some(oi))
            .expect("top_k");
        dist.release_into(&mut pool);
        start += rows;
    }
    if let Some(yn) = ynorm {
        yn.release_into(&mut pool);
    }
    // Force completion of every queued tile.
    let out: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_raw(idx_handle, n_query * k);
    let _ = out.to_host(&pool);
    t0.elapsed().as_secs_f64()
}

#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn knn_tile_size_sensitivity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let (n_train, d, n_query, k) = (20_000usize, 16usize, 4_000usize, 5usize);
    println!("n_train={n_train} d={d} n_query={n_query} k={k}");
    // Warm the JIT so the first timed row is steady-state.
    let _ = run(n_train, d, 512, k, 512, true);

    // Each row runs the SAME total work cut into `n_tiles` tiles, with the
    // training-set norm term recomputed per tile (`plain`) vs hoisted out of the
    // loop (`hoisted`).
    //
    // The ladder is walked in BOTH directions as a confound control: a real
    // per-tile cost reproduces in both orders, whereas device-memory
    // fragmentation accumulating across runs would track RUN ORDER instead and
    // the two passes would disagree.
    let ladder = [4_000usize, 1_000, 256];
    for (label, reversed) in [("ascending tile count", false), ("descending tile count", true)] {
        println!("-- {label} --");
        println!(
            "{:>8} {:>7} {:>12} {:>12} {:>9}",
            "tile", "n_tiles", "plain_s", "hoisted_s", "speedup"
        );
        let mut tiles: Vec<usize> = ladder.to_vec();
        if reversed {
            tiles.reverse();
        }
        for &tile in &tiles {
            let plain = run(n_train, d, n_query, k, tile, false);
            let hoisted = run(n_train, d, n_query, k, tile, true);
            println!(
                "{tile:>8} {:>7} {plain:>12.4} {hoisted:>12.4} {:>8.1}x",
                n_query.div_ceil(tile),
                plain / hoisted.max(1e-9)
            );
        }
    }
}
