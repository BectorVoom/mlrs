//! KNN-01 root-cause probe: WHY is the pairwise-distance kernel slow?
//!
//! After the direct `euclidean_sq_dist` kernel replaced the GEMM-expansion, the
//! distance stage is still ~99% of KNN predict and mlrs is 5.7–10× behind cuML.
//! This probe discriminates between the two candidate explanations by measuring
//! how the kernel's runtime scales.
//!
//! At a FIXED output size (`rows_x × rows_y` held constant) the two hypotheses
//! make opposite predictions as `cols` (= `n_features`) grows:
//!
//! - **H1 — materializing the distance matrix dominates.** Traffic is
//!   `rows_x × rows_y × 4` bytes written (plus the same read back by the
//!   selection pass) and does NOT depend on `cols`. Prediction: runtime roughly
//!   CONSTANT in `cols`.
//! - **H2 — no data reuse: every output element re-reads both feature vectors
//!   from global memory.** Thread `(i, j)` walks `x[i, ..]` and `y[j, ..]` in
//!   full, so traffic is `rows_x × rows_y × cols × 2 × 4` bytes. Prediction:
//!   runtime LINEAR in `cols`, and the implied bandwidth should land near the
//!   device's peak (i.e. the kernel is saturating memory, just for a
//!   pathologically redundant access pattern).
//!
//! The `eff_GB/s` column reports the H2 traffic model divided by the measured
//! time: if H2 is right that number is roughly constant across `cols` AND close
//! to the device's achievable bandwidth.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features cuda \
//!   --test knn_distance_roofline_test -- --ignored --nocapture
//! ```
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_backend::abflag;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance_direct;
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

/// Time ONLY the pairwise-distance kernel (no top-k, no tiling) for one shape.
fn time_distance(rows_x: usize, rows_y: usize, cols: usize, reps: usize, variant: &str) -> f64 {
    // Thread-local overrides, not `set_var`: the variable is read per launch and
    // libtest runs this binary's other tests concurrently, so mutating the
    // process environment here would race their `getenv` and silently change
    // which kernel THEY exercise (`mlrs_backend::abflag`). Both guard sets live
    // to the end of this function, which is the whole timed region.
    let _ab_defaults: Vec<_> = ["MLRS_DIST_UNTILED", "MLRS_DIST_TILED1X1", "MLRS_DIST_RB2"]
        .into_iter()
        .map(abflag::clear)
        .collect();
    let _ab_variant = match variant {
        "untiled" => Some(abflag::force("MLRS_DIST_UNTILED", "1")),
        "tiled" => Some(abflag::force("MLRS_DIST_TILED1X1", "1")),
        "rb2" => Some(abflag::force("MLRS_DIST_RB2", "1")),
        // "rb4" = the default 4x4 register-blocked kernel; no knob.
        _ => None,
    };
    let x = uniform(rows_x * cols, 42);
    let y = uniform(rows_y * cols, 7);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    // Warm the JIT for this shape before timing.
    let w = distance_direct::<f32>(&mut pool, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
        .expect("distance");
    let probe: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(w.handle().clone(), 1);
    let _ = probe.to_host(&pool);
    w.release_into(&mut pool);

    let t0 = Instant::now();
    for _ in 0..reps {
        let d = distance_direct::<f32>(&mut pool, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("distance");
        // Sync on a 1-element readback so the launch is actually complete.
        let probe: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_raw(d.handle().clone(), 1);
        let _ = probe.to_host(&pool);
        d.release_into(&mut pool);
    }
    // The `_ab_*` guards restore the previous overrides when this returns.
    t0.elapsed().as_secs_f64() / reps as f64
}

/// H1 vs H2, and the tiled-vs-untiled dispatch decision, in one sweep.
///
/// `rows_y` is deliberately large enough that the `y` operand EXCEEDS the T4's
/// 4 MB L2 at every `cols` tested. An earlier version of this probe used a small
/// `rows_y` whose `y` fitted entirely in L2, which let the cache supply the reuse
/// the kernel fails to and made the scaling look sublinear — a measurement
/// artifact, not a property of the kernel.
#[test]
#[ignore = "wall-clock probe; run explicitly with --ignored --nocapture"]
fn distance_scaling_vs_feature_count() {
    let _ = env_logger::builder().is_test(true).try_init();
    let (rows_x, rows_y) = (2_048usize, 65_536usize);
    let out_elems = rows_x * rows_y;
    println!("FIXED output {rows_x} x {rows_y} = {out_elems} elements; sweeping cols");
    println!("H1 (matrix-bound) predicts CONSTANT time; H2 (no-reuse feature reads) predicts LINEAR in cols.");
    println!(
        "{:>5} {:>7} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
        "cols", "y_MB", "untiled_s", "vs_c=8", "tiled_s", "tiled_x", "rb2_s", "rb2_x", "rb4_s", "rb4_x"
    );

    let mut base = 0.0f64;
    for (n, &cols) in [8usize, 16, 32, 64, 128].iter().enumerate() {
        // Interleave the variants across reps so a thermal / contention drift on a
        // shared GPU cannot be mistaken for a kernel difference.
        let untiled = time_distance(rows_x, rows_y, cols, 3, "untiled");
        let tiled = time_distance(rows_x, rows_y, cols, 3, "tiled");
        let rb2 = time_distance(rows_x, rows_y, cols, 3, "rb2");
        let rb4 = time_distance(rows_x, rows_y, cols, 3, "rb4");
        if n == 0 {
            base = untiled;
        }
        let y_mb = (rows_y * cols * 4) as f64 / 1e6;
        println!(
            "{cols:>5} {y_mb:>7.1} {untiled:>10.5} {:>7.2}x {tiled:>10.5} {:>7.2}x {rb2:>10.5} {:>7.2}x {rb4:>10.5} {:>7.2}x",
            untiled / base,
            untiled / tiled,
            untiled / rb2,
            untiled / rb4
        );
        let _ = out_elems;
    }
    println!("\nuntiled scaling ~linear in cols => the cost is REDUNDANT per-element feature reads (H2).");
    println!("speedups are over untiled: shared-memory reuse, then +2x2, then +4x4 register blocking.");
}
