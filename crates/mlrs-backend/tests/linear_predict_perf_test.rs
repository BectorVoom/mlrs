//! `prims::linear_predict` wall-clock probe — the LINEAR-PRED-CPU lever.
//!
//! A plain `std::time::Instant` probe (the `linear_regression_perf_test.rs` /
//! `kmeans_perf_test.rs` precedent — NOT a Criterion benchmark), `#[ignore]`d so
//! the ordinary suite stays fast. It times the PRIM alone — no `fit`, no PyO3,
//! no Arrow ingress — so a kernel/dispatch change can be A/B'd in seconds
//! instead of paying the minutes-long cpu `fit` the estimator-level probe needs:
//!
//! ```text
//! cargo test -p mlrs-backend --release --features cpu \
//!   --test linear_predict_perf_test -- --ignored --nocapture
//! ```
//!
//! Three numbers are reported per config because they answer different
//! questions on the cpu backend (where cubecl-cpu JITs at LLVM `-O0` and gives
//! one OS thread per unit):
//!
//! * `upload` — `DeviceArray::from_host` of the `m × n` operand. On cpu this is
//!   a plain host copy and it bounds any predict that ingests fresh data.
//! * `kernel` — best-of-N steady-state `linear_predict` + terminal readback.
//! * `cold` — the FIRST `linear_predict` call, which pays the cubecl-cpu JIT. A
//!   one-shot inference process pays this, so a dispatch that reaches more
//!   distinct kernels can regress it while steady state improves.
//!
//! Compare against `scripts/bench_linear_predict_cpu.py` (sklearn, through the
//! full Python estimator API) on the same shapes.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use std::time::Instant;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::{linear_predict, linear_predict_host};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Counter-based splitmix64 (byte-identical to `linear_regression_perf_test.rs`).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_pm1(state: &mut u64) -> f32 {
    (((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
}

/// `(upload_s, best_kernel_s, cold_kernel_s, best_host_s)` for an `m × n`
/// predict. `host_s` is the zero-copy `linear_predict_host` path — the whole
/// point of the comparison is that `upload_s + kernel_s` is what the device
/// path costs on cpu, against `host_s` alone for the host path.
fn run_config(m: usize, n: usize, reps: usize) -> (f64, f64, f64, f64) {
    let mut s = 42u64;
    let x: Vec<f32> = (0..m * n).map(|_| uniform_pm1(&mut s)).collect();
    let coef: Vec<f32> = (0..n).map(|_| uniform_pm1(&mut s)).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // STEADY-STATE upload: the first `from_host` of a fresh `m·n` buffer also
    // pays first-touch page faults for the whole allocation, which a repeated
    // `predict` (whose pool hands back the same recycled buffer) does not — so
    // time it best-of-N, releasing each copy back to the pool the way the
    // estimator path does.
    let mut upload_s = f64::INFINITY;
    let mut x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    for _ in 0..reps {
        // Release BEFORE re-acquiring, exactly like consecutive `predict` calls:
        // the previous operand is back on the free list when the next upload
        // asks for its byte size, so this measures a recycled-buffer copy.
        x_dev.release_into(&mut pool);
        let t0 = Instant::now();
        x_dev = DeviceArray::from_host(&mut pool, &x);
        upload_s = upload_s.min(t0.elapsed().as_secs_f64());
    }

    let coef_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &coef);
    let bias_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[0.25f32]);

    let mut best = f64::INFINITY;
    let mut cold = 0.0;
    let mut guard = 0.0f64;
    for i in 0..reps {
        let t = Instant::now();
        let pred = linear_predict::<f32>(&mut pool, &x_dev, &coef_dev, &bias_dev, (m, n))
            .expect("linear_predict")
            .to_host(&pool);
        let el = t.elapsed().as_secs_f64();
        // Touch the result so nothing above can be optimized away.
        guard += pred[m / 2] as f64;
        if i == 0 {
            cold = el;
        }
        best = best.min(el);
    }
    let mut host_best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let pred = linear_predict_host::<f32>(&x, &coef, 0.25f32, (m, n)).expect("host predict");
        host_best = host_best.min(t.elapsed().as_secs_f64());
        assert!(pred.operand_finite, "the splitmix64 design is finite");
        guard += pred.values[m / 2] as f64;
    }

    assert!(guard.is_finite(), "degenerate predict — perf run is broken");
    (upload_s, best, cold, host_best)
}

#[test]
#[ignore = "wall-clock perf probe — run with --release --ignored --nocapture"]
fn linear_predict_perf_ladder() {
    // The predict shapes `scripts/bench_linear_predict_cpu.py` compares against
    // sklearn, spanning the small-`n` (bandwidth-bound) and `n = 64` (the
    // fitted `GRAM_EIG_MAX_FEATURES` ceiling) ends of the feature axis.
    let configs: &[(usize, usize)] = &[
        (1_000, 16),
        (5_000, 16),
        (10_000, 16),
        (50_000, 16),
        (100_000, 16),
        (100_000, 64),
        (1_000_000, 16),
        (200_000, 64),
    ];
    println!(
        "{:>9} {:>4} | {:>10} {:>10} {:>10} | {:>10} {:>8} {:>8}",
        "m", "n", "upload (s)", "kernel (s)", "cold (s)", "host (s)", "GB/s", "speedup"
    );
    for &(m, n) in configs {
        let (up, best, cold, host) = run_config(m, n, 6);
        let gbs = (m * n * 4) as f64 / host / 1e9;
        let speedup = (up + best) / host;
        println!(
            "{m:>9} {n:>4} | {up:>10.4} {best:>10.4} {cold:>10.4} | \
             {host:>10.4} {gbs:>8.2} {speedup:>7.2}x"
        );
    }
}
