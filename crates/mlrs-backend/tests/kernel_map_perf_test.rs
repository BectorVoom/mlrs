//! In-place kernel-map launch geometry — the LAUNCH, isolated.
//!
//! `kernel_matrix` spends almost all of its time in the base op (`gemm` /
//! `distance`, `O(n²·d)`) and only `O(n²)` in the per-element map, so timing a
//! whole `kernel_matrix` call is a poor instrument for a change to the MAP's
//! launch geometry — the base op would dilute it away. This probe launches
//! `rbf_map` alone over a resident buffer, under each cube width, so the
//! scheduling choice shows up undiluted.
//!
//! ```text
//! cargo test -p mlrs-backend --release --features cpu \
//!   --test kernel_map_perf_test -- --ignored --nocapture
//! ```
//!
//! What is actually being swept: on the CPU runtime a `cube_dim` IS an OS
//! thread count (`cubecl-cpu` maps one thread per unit and runs the cube grid
//! as a serial loop inside each), so the GPU-idiomatic 256 spawns 256 threads
//! and pays a 256-way join per launch. On a GPU backend the width is a
//! warp/wavefront multiple and 256 is the right answer, which is exactly what
//! `capability::gather_launch_width()` encodes — this probe measures whether it
//! is, on the machine in front of you, rather than assuming it.
//!
//! `MLRS_KMAP_N` overrides the element count; `MLRS_KMAP_WIDTHS` the swept
//! widths; `MLRS_KMAP_REPS` the repetition count.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::{launch_dims_1d, launch_dims_1d_folded};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_kernels::rbf_map;

/// One drained in-place `rbf_map` launch over `base` at cube width `block`, in
/// seconds.
///
/// The `block_on(client.sync())` is the drain: a launch only ENQUEUES, so timing
/// without a blocking sync measures enqueue cost and nothing else. Draining with
/// a `to_host` instead would fold an `n`-element read-back into every lap, which
/// at these sizes is larger than the launch being measured.
fn map_launch(
    pool: &mut BufferPool<ActiveRuntime>,
    base: &DeviceArray<ActiveRuntime, f32>,
    n: usize,
    block: u32,
) -> f64 {
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d_folded(n, block);
    let t0 = Instant::now();
    // SAFETY: `n` is the buffer's own element count and `rbf_map` bounds-checks
    // `tid < input.len()`; in and out are the SAME handle (the in-place map
    // idiom `launch_map_in_place` uses).
    let in_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    rbf_map::launch::<f32, ActiveRuntime>(&client, count, dim, in_arg, out_arg, 0.0f32);
    let _ = cubecl::future::block_on(client.sync());
    t0.elapsed().as_secs_f64()
}

fn best_of(
    reps: usize,
    pool: &mut BufferPool<ActiveRuntime>,
    base: &DeviceArray<ActiveRuntime, f32>,
    n: usize,
    block: u32,
) -> f64 {
    // One untimed warm-up: the first launch of a shape pays the pipeline
    // compile, which is a one-off and not what this probe is about.
    map_launch(pool, base, n, block);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        best = best.min(map_launch(pool, base, n, block));
    }
    best
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn kernel_map_launch_width_sweep() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    let reps: usize = std::env::var("MLRS_KMAP_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let widths: Vec<u32> = std::env::var("MLRS_KMAP_WIDTHS")
        .unwrap_or_else(|_| "1,4,8,16,32,64,128,256".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    // Pairwise element counts: the `n × n` Gram sizes `KernelRidge` /
    // `SpectralClustering` / `KernelDensity` actually hand the map.
    let sides: Vec<usize> = std::env::var("MLRS_KMAP_N")
        .unwrap_or_else(|_| "256,512,1024,2048".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();

    let shipped = capability::gather_launch_width();
    println!("backend={backend} reps={reps} gather_launch_width()={shipped}");
    print!("{:>7} | {:>10}", "n(side)", "elems");
    for w in &widths {
        print!(" | {:>8}", format!("w={w}"));
    }
    println!(" | {:>9} | {:>8}", "shipped", "vs 256");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for side in sides {
        let n = side * side;
        let host: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01).collect();
        // Uploaded ONCE per rung and left resident: the probe times the LAUNCH,
        // and a per-lap upload would be the only thing it measured.
        let base: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &host);

        print!("{side:>7} | {n:>10}");
        let mut at_256 = f64::NAN;
        for w in &widths {
            let t = best_of(reps, &mut pool, &base, n, *w);
            if *w == 256 {
                at_256 = t;
            }
            print!(" | {:>8.3}", t * 1e3);
        }
        let ship = best_of(reps, &mut pool, &base, n, shipped);
        println!(" | {:>9.3} | {:>7.2}x", ship * 1e3, at_256 / ship);
        base.release_into(&mut pool);
    }
}

/// Device-level witness that the FOLDED map geometry covers a pairwise element
/// count past `MAX_GRID_DIM · block`, at the size the estimators actually reach.
///
/// Measured, on this change, for what the UNFOLDED geometry the map used to use
/// does at the same point (`n = 4097² = 16 785 409`, width 256 ⇒ a single-axis
/// `[65569, 1, 1]`):
///
/// | backend | outcome |
/// |---|---|
/// | wgpu | **hard failure** — `wgpu error: Validation Error … Each current dispatch group size dimension ([65569, 1, 1]) must be less or equal to 65535` |
/// | rocm | works — HIP's grid-X limit is `2³¹−1`, so the 65 535 cap is not enforced |
/// | cpu  | works — the runtime imposes no such cap |
///
/// So 65 535 is the portability FLOOR (it is what wgpu's
/// `maxComputeWorkgroupsPerDimension` guarantees), not a universal limit, and
/// wgpu is a primary correctness gate for this repo. The unfolded launch is NOT
/// re-attempted here: on wgpu it aborts the process, which is not something a
/// test file should do to the suite it lives in.
#[test]
#[ignore = "heavy ~16.8M-element launch — run with --release --ignored --nocapture"]
fn folded_launch_covers_past_grid_cap() {
    let backend = capability::active_backend_name();
    let side = 4097usize;
    let n = side * side;
    let block = 256u32;

    let (folded, _) = launch_dims_1d_folded(n, block);
    let (unfolded, _) = launch_dims_1d(n, block);
    println!("backend={backend} n={n} block={block}\n  folded  ={folded:?}\n  unfolded={unfolded:?} (NOT launched — see the doc comment)");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client.clone());
    // A buffer of zeros: `rbf_map` at gamma = 0 maps every element to
    // exp(0) = 1, so any element still reading 0.0 afterwards was never visited.
    let host = vec![0.0f32; n];
    let base: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &host);

    let (count, dim) = launch_dims_1d_folded(n, block);
    // SAFETY: `n` is the buffer's own element count and `rbf_map` bounds-checks
    // `tid < input.len()`; in and out are the SAME handle (the in-place idiom).
    let in_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    rbf_map::launch::<f32, ActiveRuntime>(&client, count, dim, in_arg, out_arg, 0.0f32);
    let _ = cubecl::future::block_on(client.sync());

    let got = base.to_host(&pool);
    let unvisited = got.iter().filter(|v| **v == 0.0).count();
    assert_eq!(
        unvisited,
        0,
        "the folded geometry must visit every element (backend={backend}, \
         first unvisited at {:?})",
        got.iter().position(|v| *v == 0.0)
    );
    println!("  FOLDED: all {n} elements visited; last={:e}", got[n - 1]);
    base.release_into(&mut pool);
}

/// What the adapter reports as its cube-count limit, and what CubeCL's own
/// helper does with a count that exceeds it.
///
/// This is the fact that decides whether a hand-rolled `div_ceil` into
/// `CubeCount::Static(n, 1, 1)` is safe: it is not, because 65 535 is only
/// wgpu's floor and the real limit is per-adapter. `calculate_cube_count_elemwise`
/// consults `hardware.max_cube_count` and SPREADS across axes, so it is the
/// portable answer and a hand-rolled count is not.
#[test]
#[ignore = "backend probe — run with --release --ignored --nocapture"]
fn adapter_cube_count_limit_probe() {
    let backend = capability::active_backend_name();
    let client = runtime::active_client();
    let max = client.properties().hardware.max_cube_count;
    let dim = CubeDim { x: 256, y: 1, z: 1 };
    let n = 4097usize * 4097; // 16 785 409 -> 65 569 cubes at width 256

    let framework = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    let hand_rolled = launch_dims_1d(n, 256).0;
    let folded = launch_dims_1d_folded(n, 256).0;

    println!(
        "backend={backend} hardware.max_cube_count={max:?}\n  \
         n={n} at cube_dim.x=256 needs {} cubes\n  \
         calculate_cube_count_elemwise -> {framework:?}\n  \
         hand-rolled div_ceil         -> {hand_rolled:?}\n  \
         mlrs launch_dims_1d_folded   -> {folded:?}",
        n.div_ceil(256)
    );
}
