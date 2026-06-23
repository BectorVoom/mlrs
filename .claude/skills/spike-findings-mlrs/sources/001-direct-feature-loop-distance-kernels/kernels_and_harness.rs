//! SPIKE 001 (Phase 13 keystone) — TEMPORARY run vehicle, not production code.
//!
//! Proves the genuinely-new cpu-MLIR unknown named in decision D-06: a **direct
//! pairwise distance kernel that loops over the feature dimension** computing an
//! accumulator with `.abs()`, a running max, and in-kernel `F::powf` — as ONE
//! `#[cube(launch)]` kernel — launches under `--features cpu` and matches a host
//! reference. Existing direct kernels (rbf/poly/sigmoid) are elementwise over a
//! *precomputed* matrix; none loop over features. This isolates that risk.
//!
//! Durable artifact: copied to `.planning/spikes/001-direct-feature-loop-distance-kernels/`.
//! Delete this file after the spike is recorded (it is not part of the real prim).
//!
//! Run: `cargo test -p mlrs-backend --features cpu --test knn_spike_001_test -- --nocapture`

use cubecl::prelude::*;
use mlrs_backend::runtime::{self, ActiveRuntime};

// ─────────────────────────────────────────────────────────────────────────────
// Candidate direct pairwise distance kernels (cpu-MLIR-safe idiom):
//   one unit per output element (i, j); a runtime `while kk < cols` loop over the
//   feature dim; only `F`/`u32` accumulators + `if` guards; no SharedMemory, no
//   mutable bool, no `F::INFINITY`, no descending-shift loop. `.abs()` is the
//   jacobi-proven instance form; `F::powf` is the poly_map-proven STATIC form.
//   out is row-major (rows_x × rows_y): out[i*rows_y + j].
// ─────────────────────────────────────────────────────────────────────────────

#[cube(launch)]
pub fn manhattan_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += diff;
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

#[cube(launch)]
pub fn chebyshev_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            // running max via STATEMENT-form `if` (epanechnikov-proven); diffs are
            // ≥0 so seeding acc at 0 is correct.
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                if diff > acc {
                    acc = diff;
                }
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

#[cube(launch)]
pub fn minkowski_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    p: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            // THE NAMED UNKNOWN: in-kernel `F::powf` inside a feature-loop
            // accumulator, then a final `^(1/p)` root.
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += F::powf(diff, p);
                kk += 1u32;
            }
            let inv_p = F::new(1.0) / p;
            out[(i * rows_y + j) as usize] = F::powf(acc, inv_p);
        }
    }
}

/// 2D launch config: i (rows_x) on X, j (rows_y) on Y, 16×16 cube, ceiling div.
fn launch_2d(rows_x: usize, rows_y: usize) -> (CubeCount, CubeDim) {
    let b = 16u32;
    let cx = ((rows_x as u32) + b - 1) / b;
    let cy = ((rows_y as u32) + b - 1) / b;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: b, y: b, z: 1 },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-precision harness. Kernels are generic over F (mandate); the host test is
// instantiated concretely for f64 (the cpu gate precision) and f32.
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! spike001 {
    ($name:ident, $F:ty, $tol:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::builder().is_test(true).try_init();
            let client = runtime::active_client();

            // Small fixed X (n=5, d=3), self-pairwise (Y = X) — the KNN-graph case.
            // Includes a duplicate row (rows 0 and 4 equal) to exercise dist-0 ties.
            let n = 5usize;
            let d = 3usize;
            let x: Vec<$F> = vec![
                0.0, 0.0, 0.0, // 0
                1.0, 2.0, 3.0, // 1
                -2.0, 0.5, 4.0, // 2
                3.0, -1.0, 0.0, // 3
                0.0, 0.0, 0.0, // 4 (duplicate of 0)
            ];
            let p: $F = 3.0; // Minkowski-3 (a non-trivial, non-integer-friendly root)

            // Host references (same precision as the device path).
            let mut ref_manhattan = vec![0 as $F; n * n];
            let mut ref_chebyshev = vec![0 as $F; n * n];
            let mut ref_minkowski = vec![0 as $F; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut l1 = 0 as $F;
                    let mut linf = 0 as $F;
                    let mut lp = 0 as $F;
                    for kk in 0..d {
                        let diff = (x[i * d + kk] - x[j * d + kk]).abs();
                        l1 += diff;
                        if diff > linf {
                            linf = diff;
                        }
                        lp += diff.powf(p);
                    }
                    ref_manhattan[i * n + j] = l1;
                    ref_chebyshev[i * n + j] = linf;
                    ref_minkowski[i * n + j] = lp.powf(1.0 / p);
                }
            }

            let run = |kernel: &str| -> Vec<$F> {
                let xh = client.create(cubecl::bytes::Bytes::from_elems(x.clone()));
                let yh = client.create(cubecl::bytes::Bytes::from_elems(x.clone()));
                let oh = client.empty(n * n * std::mem::size_of::<$F>());
                let o_read = oh.clone();
                let (count, dim) = launch_2d(n, n);
                // cubecl 0.10: 2-arg by-value form `from_raw_parts(handle, len)`
                // (consumes the handle) — see spike_test.rs / distance.rs.
                let xa = unsafe { ArrayArg::from_raw_parts(xh, n * d) };
                let ya = unsafe { ArrayArg::from_raw_parts(yh, n * d) };
                let oa = unsafe { ArrayArg::from_raw_parts(oh, n * n) };
                match kernel {
                    "manhattan" => manhattan_dist::launch::<$F, ActiveRuntime>(
                        &client, count, dim, xa, ya, oa, n as u32, n as u32, d as u32,
                    ),
                    "chebyshev" => chebyshev_dist::launch::<$F, ActiveRuntime>(
                        &client, count, dim, xa, ya, oa, n as u32, n as u32, d as u32,
                    ),
                    "minkowski" => minkowski_dist::launch::<$F, ActiveRuntime>(
                        &client, count, dim, xa, ya, oa, n as u32, n as u32, d as u32, p,
                    ),
                    _ => unreachable!(),
                }
                let bytes = client.read_one(o_read).expect("read-back out");
                bytemuck::cast_slice::<u8, $F>(&bytes).to_vec()
            };

            let check = |got: &[$F], want: &[$F], label: &str| {
                for idx in 0..(n * n) {
                    let diff = (got[idx] - want[idx]).abs();
                    assert!(
                        (diff as f64) <= $tol,
                        "{label}[{idx}] device={} host={} |Δ|={}",
                        got[idx],
                        want[idx],
                        diff
                    );
                }
                println!("  {label}: OK ({} elems, tol {})", n * n, $tol);
            };

            println!("SPIKE 001 [{}]:", stringify!($F));
            check(&run("manhattan"), &ref_manhattan, "manhattan");
            check(&run("chebyshev"), &ref_chebyshev, "chebyshev");
            check(&run("minkowski"), &ref_minkowski, "minkowski-p(3)");
            // Duplicate-row sanity: d(0,4) must be exactly 0 for every metric.
            let m = run("minkowski");
            assert_eq!(m[0 * n + 4] as f64, 0.0, "dup-row Minkowski self-dist != 0");
        }
    };
}

spike001!(spike001_f64_direct_distance_kernels, f64, 1e-6);
spike001!(spike001_f32_direct_distance_kernels, f32, 1e-3);

/// DEPTH PROBE (f64): does the general Minkowski-p kernel SUBSUME the special
/// metrics — Minkowski(1) == Manhattan, Minkowski(2) == true Euclidean — and does
/// a genuinely non-integer exponent (p=1.5) lower correctly through `F::powf`?
/// Answers the named Claude's-discretion decision: special-case p∈{1,2} to fast
/// paths, or route everything through one general kernel?
#[test]
fn spike001_minkowski_subsumes_l1_l2_and_handles_fractional_p() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let n = 4usize;
    let d = 3usize;
    let x: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0, -2.0, 0.5, 4.0, 3.0, -1.0, 0.0,
    ];

    let run_mink = |p: f64| -> Vec<f64> {
        let xh = client.create(cubecl::bytes::Bytes::from_elems(x.clone()));
        let yh = client.create(cubecl::bytes::Bytes::from_elems(x.clone()));
        let oh = client.empty(n * n * std::mem::size_of::<f64>());
        let o_read = oh.clone();
        let (count, dim) = launch_2d(n, n);
        let xa = unsafe { ArrayArg::from_raw_parts(xh, n * d) };
        let ya = unsafe { ArrayArg::from_raw_parts(yh, n * d) };
        let oa = unsafe { ArrayArg::from_raw_parts(oh, n * n) };
        minkowski_dist::launch::<f64, ActiveRuntime>(
            &client, count, dim, xa, ya, oa, n as u32, n as u32, d as u32, p,
        );
        let bytes = client.read_one(o_read).expect("read-back");
        bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
    };

    // Host L1 / L2 / Minkowski(1.5) references.
    let (mut l1, mut l2, mut m15) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    for i in 0..n {
        for j in 0..n {
            let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
            for kk in 0..d {
                let diff = (x[i * d + kk] - x[j * d + kk]).abs();
                a += diff;
                b += diff * diff;
                c += diff.powf(1.5);
            }
            l1[i * n + j] = a;
            l2[i * n + j] = b.sqrt();
            m15[i * n + j] = c.powf(1.0 / 1.5);
        }
    }

    let (mk1, mk2, mk15) = (run_mink(1.0), run_mink(2.0), run_mink(1.5));
    for idx in 0..(n * n) {
        assert!((mk1[idx] - l1[idx]).abs() <= 1e-9, "Mink(1)!=L1 @{idx}");
        assert!((mk2[idx] - l2[idx]).abs() <= 1e-9, "Mink(2)!=L2(euclid) @{idx}");
        assert!((mk15[idx] - m15[idx]).abs() <= 1e-9, "Mink(1.5) wrong @{idx}");
    }
    println!(
        "SPIKE 001 depth: Minkowski(1)==Manhattan ✓  Minkowski(2)==Euclidean ✓  \
         Minkowski(1.5) fractional-p ✓  → one general kernel can subsume L1/L2 \
         (fast-path special-casing is an optimization choice, not a correctness need)"
    );
}
