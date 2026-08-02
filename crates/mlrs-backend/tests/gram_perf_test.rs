//! Gram/Xty formation wall-clock probe — the PHASE, isolated.
//!
//! `ridge_positive_perf_test.rs` times a whole `Ridge(positive=True)` fit, which
//! is the number that matters but is a poor instrument for a kernel change: the
//! design upload and the solve sit in the same lap. This probe times ONLY
//! `column_means` + `gram_xty_centered` — the pair the `positive` arm spends
//! almost all of its device time in — with the operands already resident, so a
//! kernel edit shows up undiluted.
//!
//! ```text
//! cargo test -p mlrs-backend --release --features wgpu \
//!   --test gram_perf_test -- --ignored --nocapture
//! ```
//!
//! Sweeps the two formations and the tiled kernel's cube width, because the
//! cube shape is an ADAPTER property (warp width, register file, scheduler)
//! and cannot be derived — it has to be swept on the machine that will run it.
//! `MLRS_GRAM_DIMS` overrides the swept widths (comma-separated); `MLRS_GRAM_N`
//! the row count.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram::{column_means, gram_xty_centered};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// One drained `column_means` + `gram_xty_centered` pass over a resident
/// design, in seconds.
///
/// The `to_host` of the length-1 `ymean` is the drain: `client.sync()` returns
/// a FUTURE, so timing without a real blocking read-back measures enqueue time
/// and nothing else (the RIDGE-POS-PERF finding).
fn gram_phase(pool: &mut BufferPool<ActiveRuntime>, x: &DeviceArray<ActiveRuntime, f32>, y: &DeviceArray<ActiveRuntime, f32>, n: usize, d: usize) -> f64 {
    let t0 = Instant::now();
    let (xm, ym) = column_means::<f32>(pool, x, y, n, d).expect("column_means");
    let (gram, xty) = gram_xty_centered::<f32>(pool, x, y, (&xm, &ym), n, d).expect("gram");
    let probe = gram.to_host(pool);
    let secs = t0.elapsed().as_secs_f64();
    assert!(probe[0].is_finite(), "degenerate gram at n={n} d={d}");
    xm.release_into(pool);
    ym.release_into(pool);
    gram.release_into(pool);
    xty.release_into(pool);
    secs
}

fn best_of(
    reps: usize,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, f32>,
    y: &DeviceArray<ActiveRuntime, f32>,
    n: usize,
    d: usize,
) -> f64 {
    gram_phase(pool, x, y, n, d);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        best = best.min(gram_phase(pool, x, y, n, d));
    }
    best
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn gram_formation_perf_sweep() {
    if mlrs_backend::capability::active_backend_name() == "cpu" {
        println!("gram_perf backend=cpu: SKIPPED (gram_path is the gemm arm there)");
        return;
    }
    let reps: usize = std::env::var("MLRS_POS_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let n: usize = std::env::var("MLRS_GRAM_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let dims: Vec<u32> = std::env::var("MLRS_GRAM_DIMS")
        .unwrap_or_else(|_| "32,64,128,256".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    println!(
        "backend={} n={n} reps={reps}",
        mlrs_backend::capability::active_backend_name()
    );
    print!("{:>5} | {:>10}", "d", "blocked");
    for w in &dims {
        print!(" | {:>8}", format!("t/{w}"));
    }
    println!(" | {:>8} {:>7}", "t/auto", "best");

    for d in [16usize, 32, 64, 128, 256] {
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i % 251) as f32) * 0.01 - 1.25 + (i % d) as f32 * 0.1)
            .collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.02 - 1.0).collect();
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

        let blocked = {
            let _g = mlrs_backend::abflag::force("LR_GRAM_BLOCKED", "1");
            best_of(reps, &mut pool, &x_dev, &y_dev, n, d)
        };
        print!("{d:>5} | {:>10.3}", blocked * 1e3);

        let mut best_dim = f64::INFINITY;
        for w in &dims {
            let _g0 = mlrs_backend::abflag::force("LR_GRAM_TILED", "1");
            let _g = mlrs_backend::abflag::force("MLRS_GRAM_TILE_DIM", &w.to_string());
            let t = best_of(reps, &mut pool, &x_dev, &y_dev, n, d);
            best_dim = best_dim.min(t);
            print!(" | {:>8.3}", t * 1e3);
        }

        let auto = {
            let _g0 = mlrs_backend::abflag::clear("LR_GRAM_TILED");
            let _g1 = mlrs_backend::abflag::clear("LR_GRAM_BLOCKED");
            let _g = mlrs_backend::abflag::clear("MLRS_GRAM_TILE_DIM");
            best_of(reps, &mut pool, &x_dev, &y_dev, n, d)
        };
        println!(
            " | {:>8.3} {:>6.2}x",
            auto * 1e3,
            blocked / best_dim.min(auto)
        );

        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
    }
}
