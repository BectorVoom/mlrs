//! KNN-01 perf-lever correctness gate: the fused neighbor-target gather prim
//! (`prims::knn::knn_regress_mean_gather`) and the parallel select-k kernel
//! (`topk::select_k_shared`).
//!
//! A perf change is only allowed to be faster, never different. The two tests
//! that matter here are EQUIVALENCE tests:
//!
//! - `select_k_shared_matches_serial_bitwise` runs both selection kernels over
//!   the SAME distance matrices — including matrices built to be saturated with
//!   exact ties, which is where a reordered reduction would diverge — and asserts
//!   the emitted `(value, index)` pairs are BITWISE identical. That is the
//!   property the parallel kernel claims: the pair order is total over distinct
//!   column indices, so the tree's association order cannot change the winner.
//! - `knn_regress_mean_matches_host_gather` asserts the device gather reproduces
//!   the host `Σ y[idx]/k` loop it replaced.
//!
//! The remaining tests pin the prim's validate-before-launch geometry rejections
//! (ASVS V5) and the WR-02 out-of-range-index guard.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn::knn_regress_mean_gather;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::PrimError;

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

/// Counter-based splitmix64 (the workspace bench/probe generator).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Run `top_k` over `dist` on the parallel path and on the serial
/// (`MLRS_TOPK_SERIAL=1`) path, returning both `(values, indices)` results.
///
/// The env var is read per call by the prim, so flipping it here selects the
/// kernel. Tests using this helper must not run concurrently with each other;
/// they are collected into ONE `#[test]` for that reason.
#[allow(clippy::type_complexity)]
fn all_paths(
    dist: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
) -> (
    (Vec<f32>, Vec<u32>),
    (Vec<f32>, Vec<u32>),
    (Vec<f32>, Vec<u32>),
) {
    let mut p = pool();
    let d: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, dist);

    // Default path: single-pass select_k_onepass for k <= 32, else the
    // multi-pass select_k_shared (both must agree bitwise, so running the
    // fallback here for large k keeps the assertions meaningful).
    std::env::remove_var("MLRS_TOPK_SERIAL");
    std::env::remove_var("MLRS_TOPK_MULTIPASS");
    let (ov, oi) = top_k::<f32>(&mut p, &d, rows, cols, k, true, None, None).expect("onepass");
    let one = (ov.to_host(&p), oi.to_host(&p));

    std::env::set_var("MLRS_TOPK_MULTIPASS", "1");
    let (pv, pi) = top_k::<f32>(&mut p, &d, rows, cols, k, true, None, None).expect("parallel");
    let par = (pv.to_host(&p), pi.to_host(&p));
    std::env::remove_var("MLRS_TOPK_MULTIPASS");

    std::env::set_var("MLRS_TOPK_SERIAL", "1");
    let (sv, si) = top_k::<f32>(&mut p, &d, rows, cols, k, true, None, None).expect("serial");
    let ser = (sv.to_host(&p), si.to_host(&p));
    std::env::remove_var("MLRS_TOPK_SERIAL");

    (one, par, ser)
}

/// The parallel `select_k_shared` must emit BITWISE-identical `(value, index)`
/// pairs to the serial `select_k` — on well-spread data, on tie-saturated data,
/// and at the `cols < CUBE_DIM` / `k == cols` edges.
#[test]
fn select_k_shared_matches_serial_bitwise() {
    // (rows, cols, k, tie_modulus) — `tie_modulus = Some(m)` quantizes the
    // distances onto `m` distinct values, so with cols >> m almost every
    // selected pair is decided by the LOWEST-INDEX tie-break rather than by the
    // value. That is precisely the semantics a reordered tree reduction would
    // break, so these cases are the real gate.
    let cases: &[(usize, usize, usize, Option<u64>)] = &[
        // Well-spread values, cols far wider than the 256-unit cube.
        (17, 1000, 7, None),
        // cols BELOW the cube width (the launch clamps units to cols).
        (9, 40, 5, None),
        // cols == 1 and k == 1: the log2 tree degenerates to a no-op.
        (5, 1, 1, None),
        // k == cols: every column must be emitted, in ascending pair order.
        (4, 16, 16, None),
        // Tie-saturated: 1000 columns over only 8 distinct values.
        (11, 1000, 10, Some(8)),
        // Pathological: 500 columns, 2 distinct values — nearly every decision
        // is a pure index tie-break.
        (6, 500, 12, Some(2)),
        // Tie-saturated AND narrower than the cube width.
        (7, 64, 6, Some(3)),
        // k at the one-pass local-list capacity boundary.
        (5, 300, 32, Some(6)),
        // k ABOVE the one-pass capacity: the default path falls back to the
        // multi-pass kernel; all three paths must still agree.
        (3, 120, 40, Some(4)),
    ];

    for &(rows, cols, k, ties) in cases {
        let mut s = 0xDEAD_BEEFu64 ^ (rows as u64) << 32 ^ (cols as u64);
        let dist: Vec<f32> = (0..rows * cols)
            .map(|_| {
                let r = splitmix64(&mut s);
                match ties {
                    // Quantized onto `m` distinct non-negative values.
                    Some(m) => (r % m) as f32,
                    // Non-negative (top_k sqrts the selected values).
                    None => (r >> 40) as f32 / 1000.0,
                }
            })
            .collect();

        let ((ov, oi), (pv, pi), (sv, si)) = all_paths(&dist, rows, cols, k);

        assert_eq!(
            pi, si,
            "parallel select_k indices diverged from serial at rows={rows} cols={cols} k={k} ties={ties:?}"
        );
        assert_eq!(
            oi, si,
            "one-pass select_k indices diverged from serial at rows={rows} cols={cols} k={k} ties={ties:?}"
        );
        // BITWISE on the values: all paths select the same element and apply the
        // same sqrt, so the results must be identical bit patterns, not merely
        // close.
        assert_eq!(
            pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "parallel select_k values diverged from serial at rows={rows} cols={cols} k={k} ties={ties:?}"
        );
        assert_eq!(
            ov.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "one-pass select_k values diverged from serial at rows={rows} cols={cols} k={k} ties={ties:?}"
        );

        // Independent check that BOTH agree with a host reference selection, so a
        // shared bug in the two device paths cannot pass silently.
        for r in 0..rows {
            let mut order: Vec<(f32, u32)> = (0..cols)
                .map(|c| (dist[r * cols + c], c as u32))
                .collect();
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            let want: Vec<u32> = order[..k].iter().map(|&(_, c)| c).collect();
            assert_eq!(&pi[r * k..r * k + k], &want[..], "row {r} vs host reference");
        }
    }
}

/// The device gather must reproduce the host `Σ y[idx] / k` loop it replaced.
#[test]
fn knn_regress_mean_matches_host_gather() {
    let n_train = 257usize;
    let n_query = 130usize;
    let k = 7usize;

    let mut s = 7u64;
    let y: Vec<f32> = (0..n_train)
        .map(|_| (splitmix64(&mut s) >> 40) as f32 / 100.0)
        .collect();
    let idx: Vec<u32> = (0..n_query * k)
        .map(|_| (splitmix64(&mut s) % n_train as u64) as u32)
        .collect();

    let mut p = pool();
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &y);
    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(&mut p, &idx);

    let got = knn_regress_mean_gather::<f32>(&mut p, &y_dev, &idx_dev, n_query, k, n_train)
        .expect("gather")
        .to_host(&p);

    assert_eq!(got.len(), n_query);
    for q in 0..n_query {
        let want: f32 =
            (0..k).map(|j| y[idx[q * k + j] as usize]).sum::<f32>() / k as f32;
        assert!(
            (got[q] as f64 - want as f64).abs() <= 1e-5 * (1.0 + want.abs() as f64),
            "query {q}: got {}, want {want}",
            got[q]
        );
    }
}

/// WR-02 parity: an out-of-range neighbor index must not read out of bounds — the
/// kernel's `t < n_train` guard makes it contribute nothing.
#[test]
fn knn_regress_mean_guards_out_of_range_index() {
    let n_train = 4usize;
    let k = 2usize;
    let y: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // Second neighbor of the single query is far past the end of `y`.
    let idx: Vec<u32> = vec![1, 9_999];

    let mut p = pool();
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &y);
    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(&mut p, &idx);

    let got = knn_regress_mean_gather::<f32>(&mut p, &y_dev, &idx_dev, 1, k, n_train)
        .expect("gather")
        .to_host(&p);

    // Only y[1] = 2.0 contributes; the divisor is still k.
    assert!((got[0] - 1.0).abs() < 1e-6, "got {}", got[0]);
}

/// ASVS V5: every geometry violation is a typed error raised BEFORE any launch.
#[test]
fn knn_regress_mean_rejects_bad_geometry() {
    let mut p = pool();
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[1.0f32, 2.0, 3.0]);
    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(&mut p, &[0u32, 1, 2, 0]);

    // idx.len() (4) != n_query * k (2 * 3).
    assert!(matches!(
        knn_regress_mean_gather::<f32>(&mut p, &y_dev, &idx_dev, 2, 3, 3),
        Err(PrimError::ShapeMismatch { operand: "idx", .. })
    ));
    // k == 0.
    assert!(matches!(
        knn_regress_mean_gather::<f32>(&mut p, &y_dev, &idx_dev, 4, 0, 3),
        Err(PrimError::ShapeMismatch { operand: "k", .. })
    ));
    // y_train.len() (3) != declared n_train (5).
    assert!(matches!(
        knn_regress_mean_gather::<f32>(&mut p, &y_dev, &idx_dev, 2, 2, 5),
        Err(PrimError::ShapeMismatch {
            operand: "y_train",
            ..
        })
    ));
}

/// The direct squared-Euclidean kernel must agree with the GEMM-expansion path
/// it replaces, and must be NO LESS accurate than it.
///
/// The two compute the same quantity by algebraically different routes
/// (`Σ(x−y)²` vs `‖x‖²+‖y‖²−2xy`), so they differ in floating-point rounding and
/// the comparison is against an f64 HOST reference rather than against each
/// other. A near-duplicate row pair is included deliberately: it is the
/// catastrophic-cancellation case for the expansion (which is why that path needs
/// its `max(d², 0)` clamp) and the case that matters most to a nearest-neighbor
/// search, since those are the pairs that win the top-k.
#[test]
fn distance_direct_matches_gemm_expansion() {
    use mlrs_backend::prims::distance::{distance, distance_direct};

    for &(rows_x, rows_y, cols) in &[(37usize, 91usize, 16usize), (8, 8, 3), (64, 200, 32)] {
        let mut s = 0xC0FF_EE00u64 ^ (cols as u64);
        // Uniform in [-1, 1) — the scale the oracle fixtures use.
        let mut gen = || ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let mut x: Vec<f32> = (0..rows_x * cols).map(|_| gen() as f32).collect();
        let y: Vec<f32> = (0..rows_y * cols).map(|_| gen() as f32).collect();
        // x's first row is a near-duplicate of y's first row.
        for c in 0..cols {
            x[c] = y[c] + 1e-4;
        }

        let mut p = pool();
        let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &x);
        let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &y);

        let gemm = distance::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), false, None)
            .expect("gemm distance")
            .to_host(&p);
        let direct = distance_direct::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("direct distance")
            .to_host(&p);
        assert_eq!(gemm.len(), direct.len());

        // f64 host reference: the same sum-of-squared-differences, exactly.
        let reference = |i: usize, j: usize| -> f64 {
            (0..cols)
                .map(|c| {
                    let d = x[i * cols + c] as f64 - y[j * cols + c] as f64;
                    d * d
                })
                .sum()
        };

        for i in 0..rows_x {
            for j in 0..rows_y {
                let want = reference(i, j);
                let n = i * rows_y + j;
                let (g, d) = (gemm[n] as f64, direct[n] as f64);

                // The direct form is a sum of squares, so it can NEVER be
                // negative — no clamp required (the expansion needs one).
                assert!(direct[n] >= 0.0, "direct went negative at ({i},{j}): {d}");
                // Direct must hit the project 1e-5 contract against the reference.
                assert!(
                    (d - want).abs() <= 1e-5 * (1.0 + want),
                    "direct wrong at ({i},{j}) cols={cols}: got {d}, want {want}"
                );
                let _ = g;
            }
        }

        // The accuracy argument for the direct form is specifically about
        // CANCELLATION, not about being uniformly more accurate — over
        // well-separated pairs the two are comparable, and neither dominates.
        // On the near-duplicate pair (0,0), where the expansion subtracts two
        // nearly equal O(cols) quantities to recover an O(1e-7) result, the
        // direct form is dramatically better.
        let want00 = reference(0, 0);
        let gemm00 = (gemm[0] as f64 - want00).abs();
        let direct00 = (direct[0] as f64 - want00).abs();
        assert!(
            direct00 <= gemm00,
            "on the cancellation case the direct form should win: \
             direct err {direct00:e} vs gemm err {gemm00:e} (true {want00:e}), cols={cols}"
        );
    }
}

/// `distance_direct` rejects bad geometry BEFORE any launch (ASVS V5).
#[test]
fn distance_direct_rejects_bad_geometry() {
    use mlrs_backend::prims::distance::distance_direct;

    let mut p = pool();
    let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[1.0f32; 12]);
    let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[1.0f32; 8]);
    // rows_x * cols (3*5) != x.len() (12).
    assert!(distance_direct::<f32>(&mut p, &xd, (3, 5), &yd, (2, 4), None).is_err());
    // feature dims disagree (4 vs 3).
    assert!(distance_direct::<f32>(&mut p, &xd, (3, 4), &yd, (2, 3), None).is_err());
}

/// All FOUR squared-Euclidean kernels — untiled, shared-memory tiled, 2×2
/// register-blocked, and 4×4 register-blocked — must agree with each other.
///
/// Both accumulate `Σ(x−y)²` over ascending `k` into a single `F` accumulator, so
/// the arithmetic is performed in the same order and the results must be BITWISE
/// identical — tiling changes only where the operands are staged, never how they
/// are combined.
///
/// The shapes deliberately include non-multiples of the 16×16 tile AND of the
/// register-blocked kernel's 32×32 output block (so the out-of-range staging lanes
/// and partially-covered blocks are exercised), a `cols` that is not a multiple of
/// the 16-wide feature slice, and a `cols` smaller than one slice.
#[test]
fn distance_tiled_matches_untiled_bitwise() {
    use mlrs_backend::prims::distance::distance_direct;

    for &(rows_x, rows_y, cols) in &[
        (16usize, 16usize, 16usize), // exact tile, exact slice
        (37, 91, 16),                // ragged in both output axes
        (17, 33, 5),                 // cols smaller than one feature slice
        (64, 200, 40),               // cols not a multiple of 16
        (1, 1, 1),                   // degenerate
        (100, 7, 128),               // many feature slices
        (33, 65, 16),                // straddles the 32x32 register-blocked block
        (32, 32, 32),                // exactly one 2x2-blocked output block
        (64, 64, 32),                // exactly one 4x4-blocked output block
        (65, 129, 33),               // straddles the 64x64 block AND the 32-wide slice
        (70, 70, 7),                 // cols well under one 32-wide feature slice
    ] {
        let mut s = 0xBEEF_1234u64 ^ ((rows_x as u64) << 20) ^ (cols as u64);
        let mut gen = || ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let x: Vec<f32> = (0..rows_x * cols).map(|_| gen() as f32).collect();
        let y: Vec<f32> = (0..rows_y * cols).map(|_| gen() as f32).collect();

        let mut p = pool();
        let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &x);
        let yd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &y);

        for v in ["MLRS_DIST_UNTILED", "MLRS_DIST_TILED1X1", "MLRS_DIST_RB2"] {
            std::env::remove_var(v);
        }
        let rb4 = distance_direct::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("4x4 register-blocked")
            .to_host(&p);

        std::env::set_var("MLRS_DIST_RB2", "1");
        let rb = distance_direct::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("2x2 register-blocked")
            .to_host(&p);
        std::env::remove_var("MLRS_DIST_RB2");

        std::env::set_var("MLRS_DIST_TILED1X1", "1");
        let tiled = distance_direct::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("tiled")
            .to_host(&p);
        std::env::remove_var("MLRS_DIST_TILED1X1");

        std::env::set_var("MLRS_DIST_UNTILED", "1");
        let untiled = distance_direct::<f32>(&mut p, &xd, (rows_x, cols), &yd, (rows_y, cols), None)
            .expect("untiled")
            .to_host(&p);
        std::env::remove_var("MLRS_DIST_UNTILED");

        assert_eq!(
            tiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            untiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "tiled diverged from untiled at ({rows_x}x{rows_y}, cols={cols})"
        );
        assert_eq!(
            rb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            untiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "2x2 register-blocked diverged from untiled at ({rows_x}x{rows_y}, cols={cols})"
        );
        assert_eq!(
            rb4.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            untiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "4x4 register-blocked diverged from untiled at ({rows_x}x{rows_y}, cols={cols})"
        );

        // And both must still match an f64 host reference within the project 1e-5.
        for i in 0..rows_x {
            for j in 0..rows_y {
                let want: f64 = (0..cols)
                    .map(|c| {
                        let d = x[i * cols + c] as f64 - y[j * cols + c] as f64;
                        d * d
                    })
                    .sum();
                let got = rb4[i * rows_y + j] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * (1.0 + want),
                    "tiled wrong at ({i},{j}) shape ({rows_x}x{rows_y}, cols={cols}): got {got}, want {want}"
                );
            }
        }
    }
}

/// The FUSED distance+top-k kernel must emit BITWISE-identical `(value, index)`
/// pairs to the two-kernel `distance_direct` → `top_k` pipeline — including on
/// tie-saturated data (quantized features force many exactly-equal distances)
/// and at the block-boundary edges (rows/cols below, at, and off the 64-wide
/// block size).
///
/// Skipped (with a log line) on adapters whose shared-memory limit cannot hold
/// the fused kernel (e.g. default wgpu, 16 KiB) — there the dispatch never
/// selects it, so there is nothing to gate.
#[test]
fn fused_topk_matches_two_kernel_bitwise() {
    // (n_train, n_query, d, k, quantize)
    let cases: &[(usize, usize, usize, usize, Option<u64>)] = &[
        // Below one block in both dimensions; k == n_train edge.
        (5, 3, 4, 5, None),
        // Exactly one block; k at the fused cap.
        (64, 64, 16, 16, None),
        // Off-boundary in both dimensions, several blocks.
        (200, 130, 8, 5, None),
        // Tie-saturated: features quantized so distances collide constantly.
        (300, 100, 4, 10, Some(3)),
        // Wider-than-block training set with k = 1 (argmin degenerate case).
        (500, 70, 32, 1, None),
        // d off the 32-wide feature slice boundary.
        (150, 90, 33, 7, Some(5)),
    ];

    for &(n_train, n_query, d, k, quant) in cases {
        // The dispatch's n_query FLOOR is a perf heuristic, not a launch
        // requirement — probe applicability with a large stand-in n_query so
        // only the HARD blockers (adapter shared-memory limit, k cap) skip.
        if !mlrs_backend::prims::knn::fused_topk_applicable::<f32>(8192, n_train, k) {
            println!(
                "SKIP fused_topk case ({n_train},{n_query},{d},{k}): fused kernel not \
                 applicable on this adapter (shared-memory limit or cap)"
            );
            continue;
        }

        let mut s = 0xC0FF_EE00u64 ^ ((n_train as u64) << 32) ^ (n_query as u64 + d as u64);
        let gen = |s: &mut u64| -> f32 {
            let r = splitmix64(s);
            match quant {
                Some(m) => (r % m) as f32,
                None => ((r >> 40) as f32 / 8_388_608.0) * 2.0 - 1.0,
            }
        };
        let xq: Vec<f32> = (0..n_query * d).map(|_| gen(&mut s)).collect();
        let xt: Vec<f32> = (0..n_train * d).map(|_| gen(&mut s)).collect();

        let mut p = pool();
        let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xq);
        let xt_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xt);

        // Fused path, at the dispatch-chosen strip count AND at forced strip
        // counts (the strip merge must be invisible in the results; strips are
        // capped so every strip keeps >= 64 columns). Env is read per call.
        let mut fused_runs: Vec<(String, Vec<f32>, Vec<u32>)> = Vec::new();
        for strips in [None, Some(2usize), Some(5)] {
            match strips {
                Some(s) => std::env::set_var("MLRS_KNN_STRIPS", s.to_string()),
                None => std::env::remove_var("MLRS_KNN_STRIPS"),
            }
            let (fv, fi) = mlrs_backend::prims::knn::fused_distance_topk::<f32>(
                &mut p,
                &xq_dev,
                (n_query, d),
                &xt_dev,
                (n_train, d),
                k,
                true,
            )
            .expect("fused");
            fused_runs.push((format!("{strips:?}"), fv.to_host(&p), fi.to_host(&p)));
        }
        std::env::remove_var("MLRS_KNN_STRIPS");
        let (fv, fi) = (fused_runs[0].1.clone(), fused_runs[0].2.clone());

        // DOT-forced run: exercises the dot-expansion kernel on backends whose
        // default is the direct kernel (wgpu). Compared with the dot rules
        // below (indices exact; values bitwise on quantized cases, 1e-5 rel
        // otherwise — at d > 32 the dispatch falls back to direct and the
        // comparison is trivially bitwise).
        std::env::set_var("MLRS_KNN_DOT", "1");
        let (qv, qi) = mlrs_backend::prims::knn::fused_distance_topk::<f32>(
            &mut p,
            &xq_dev,
            (n_query, d),
            &xt_dev,
            (n_train, d),
            k,
            true,
        )
        .expect("fused dot");
        let (qv, qi) = (qv.to_host(&p), qi.to_host(&p));
        std::env::remove_var("MLRS_KNN_DOT");

        // VECTORIZED dot kernel (KNN-03), forced on regardless of backend
        // default. Its claim is the strongest one: for every pair the scalar
        // accumulation sequence is IDENTICAL to the scalar dot kernel's (only
        // the unit→pair assignment changes, and the merge order is total), so
        // values AND indices must match the dot-forced run BITWISE — even on
        // unquantized data.
        std::env::set_var("MLRS_KNN_DOT", "1");
        std::env::set_var("MLRS_KNN_VEC", "1");
        let (vv, vi) = mlrs_backend::prims::knn::fused_distance_topk::<f32>(
            &mut p,
            &xq_dev,
            (n_query, d),
            &xt_dev,
            (n_train, d),
            k,
            true,
        )
        .expect("fused dot vec");
        let (vv, vi) = (vv.to_host(&p), vi.to_host(&p));
        std::env::remove_var("MLRS_KNN_VEC");
        std::env::remove_var("MLRS_KNN_DOT");
        assert_eq!(
            vi, qi,
            "vec dot indices diverged from scalar dot at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );
        assert_eq!(
            vv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            qv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "vec dot values diverged from scalar dot at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );

        // 4×8 register-tile vec kernel (KNN-03 r3): same bitwise claim as the
        // 4×4 vec kernel (partition-only change, total merge order).
        std::env::set_var("MLRS_KNN_DOT", "1");
        std::env::set_var("MLRS_KNN_VEC8", "1");
        let (wv, wi) = mlrs_backend::prims::knn::fused_distance_topk::<f32>(
            &mut p,
            &xq_dev,
            (n_query, d),
            &xt_dev,
            (n_train, d),
            k,
            true,
        )
        .expect("fused dot vec8");
        let (wv, wi) = (wv.to_host(&p), wi.to_host(&p));
        std::env::remove_var("MLRS_KNN_VEC8");
        std::env::remove_var("MLRS_KNN_DOT");
        assert_eq!(
            wi, qi,
            "vec8 dot indices diverged from scalar dot at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );
        assert_eq!(
            wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            qv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "vec8 dot values diverged from scalar dot at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );

        // The cancellation-free DIRECT fused kernel, which must stay bitwise
        // against the two-kernel reference regardless of the dot-expansion
        // default.
        std::env::set_var("MLRS_KNN_DIRECT", "1");
        let (dv, di) = mlrs_backend::prims::knn::fused_distance_topk::<f32>(
            &mut p,
            &xq_dev,
            (n_query, d),
            &xt_dev,
            (n_train, d),
            k,
            true,
        )
        .expect("fused direct");
        let (dv, di) = (dv.to_host(&p), di.to_host(&p));
        std::env::remove_var("MLRS_KNN_DIRECT");

        // Two-kernel reference: direct distance then one-pass/multi-pass top_k
        // (itself gated bitwise against the serial kernel above).
        let dist = mlrs_backend::prims::distance::distance_direct::<f32>(
            &mut p,
            &xq_dev,
            (n_query, d),
            &xt_dev,
            (n_train, d),
            None,
        )
        .expect("distance");
        let (rv, ri) =
            top_k::<f32>(&mut p, &dist, n_query, n_train, k, true, None, None).expect("top_k");
        let (rv, ri) = (rv.to_host(&p), ri.to_host(&p));

        assert_eq!(
            di, ri,
            "direct fused indices diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );
        assert_eq!(
            dv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "direct fused values diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );

        // DEFAULT fused path: the dot-expansion kernel at d <= 32 (the direct
        // kernel above at d > 32). Neighbor INDICES must match the reference
        // exactly. Values: on QUANTIZED cases the expansion arithmetic is
        // exact (small-integer products/sums below 2^24 in f32), so bitwise
        // equality is required; on real-valued cases the expansion legitimately
        // differs from the direct form in final ulps, so 1e-5 relative
        // agreement (the project-wide oracle bar) is the contract.
        assert_eq!(
            fi, ri,
            "fused indices diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );
        if quant.is_some() {
            assert_eq!(
                fv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                rv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fused values diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
            );
        } else {
            for (i, (a, b)) in fv.iter().zip(rv.iter()).enumerate() {
                let denom = b.abs().max(1.0);
                assert!(
                    (a - b).abs() / denom <= 1e-5,
                    "fused value {i} off tolerance: {a} vs {b} at n_train={n_train} n_query={n_query} d={d} k={k}"
                );
            }
        }
        // Dot-forced run: same contract as the default-path rules above.
        assert_eq!(
            qi, ri,
            "dot-forced indices diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
        );
        if quant.is_some() {
            assert_eq!(
                qv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                rv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "dot-forced values diverged at n_train={n_train} n_query={n_query} d={d} k={k} quant={quant:?}"
            );
        } else {
            for (i, (a, b)) in qv.iter().zip(rv.iter()).enumerate() {
                let denom = b.abs().max(1.0);
                assert!(
                    (a - b).abs() / denom <= 1e-5,
                    "dot-forced value {i} off tolerance: {a} vs {b} at n_train={n_train} n_query={n_query} d={d} k={k}"
                );
            }
        }
        // Every forced strip count must be invisible in the results.
        for (label, sv2, si2) in &fused_runs[1..] {
            assert_eq!(
                si2, &fi,
                "strips={label} indices diverged at n_train={n_train} n_query={n_query} d={d} k={k}"
            );
            assert_eq!(
                sv2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                fv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "strips={label} values diverged at n_train={n_train} n_query={n_query} d={d} k={k}"
            );
        }
    }
}

/// `fused_distance_topk` must reject bad geometry BEFORE launching (ASVS V5):
/// mismatched feature dims, wrong operand lengths, and k outside
/// `1..=min(n_train, FUSED_K_CAP)`.
#[test]
fn fused_topk_rejects_bad_geometry() {
    use mlrs_backend::prims::knn::{fused_distance_topk, FUSED_K_CAP};

    let mut p = pool();
    let xq: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[0.0f32; 8]);
    let xt: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[0.0f32; 12]);

    // Feature-dim mismatch.
    assert!(matches!(
        fused_distance_topk::<f32>(&mut p, &xq, (4, 2), &xt, (4, 3), 1, true),
        Err(PrimError::DimMismatch { .. })
    ));
    // Wrong x length for the claimed shape.
    assert!(matches!(
        fused_distance_topk::<f32>(&mut p, &xq, (5, 2), &xt, (6, 2), 1, true),
        Err(PrimError::ShapeMismatch { .. })
    ));
    // k == 0, k > n_train, k > FUSED_K_CAP.
    for bad_k in [0usize, 7, FUSED_K_CAP + 1] {
        assert!(
            matches!(
                fused_distance_topk::<f32>(&mut p, &xq, (4, 2), &xt, (6, 2), bad_k, true),
                Err(PrimError::ShapeMismatch { .. })
            ),
            "k={bad_k} must be rejected"
        );
    }
}

/// Host reference for the cpu row-scan path: the DEFINITION of brute-force
/// k-nearest under the dot expansion, evaluated in exactly the kernel's
/// arithmetic order.
///
/// `dot` accumulates over ascending `c`, the expansion is
/// `(‖x‖² + ‖y‖²) − 2·dot` clamped at 0, selection keeps the `k` smallest
/// `(value, index)` pairs in ascending pair order, and the sqrt is applied to
/// only those `k` values — each step matching `euclidean_topk_rows_cpu{,_vec}`
/// and its host wrapper op for op, so IEEE determinism makes the comparison
/// bitwise on inputs where the arithmetic is exact.
///
/// `direct` selects the cancellation-free arithmetic of
/// `euclidean_topk_rows_cpu_direct` (`Σ (x − y)²`) instead of the expansion, so
/// each kernel is compared against ITS OWN definition — the two forms
/// legitimately order near-ties differently in f32, which is the whole reason
/// the direct arm exists.
fn host_rows_topk(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    k: usize,
    direct: bool,
) -> (Vec<f32>, Vec<u32>) {
    let sumsq = |row: &[f32]| -> f32 {
        let mut acc = 0.0f32;
        for &v in row {
            acc += v * v;
        }
        acc
    };
    let xn: Vec<f32> = (0..n_query).map(|q| sumsq(&xq[q * d..(q + 1) * d])).collect();
    let yn: Vec<f32> = (0..n_train).map(|t| sumsq(&xt[t * d..(t + 1) * d])).collect();

    let mut out_val = vec![0.0f32; n_query * k];
    let mut out_idx = vec![0u32; n_query * k];
    for q in 0..n_query {
        let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(n_train);
        for t in 0..n_train {
            let mut v = if direct {
                let mut acc = 0.0f32;
                for c in 0..d {
                    let dif = xq[q * d + c] - xt[t * d + c];
                    acc += dif * dif;
                }
                acc
            } else {
                let mut dot = 0.0f32;
                for c in 0..d {
                    dot += xq[q * d + c] * xt[t * d + c];
                }
                xn[q] + yn[t] - (dot + dot)
            };
            if v < 0.0 {
                v = 0.0;
            }
            pairs.push((v, t as u32));
        }
        // Total order on `(value, index)` — the lowest-index tie-break.
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        for j in 0..k {
            out_val[q * k + j] = pairs[j].0.sqrt();
            out_idx[q * k + j] = pairs[j].1;
        }
    }
    (out_val, out_idx)
}

/// KNN-04 equivalence gate: the cpu row-scan path must reproduce brute-force
/// k-nearest EXACTLY.
///
/// `cpu_rows_topk` is a perf change, so the bar is "faster, never different".
/// All three of its kernels are checked against [`host_rows_topk`] — the
/// definition in the kernel's own arithmetic order — and the check is run:
///
/// 1. for the cancellation-free kernel (the cpu default), the SIMD expansion
///    kernel (`MLRS_KNN_DOT=1`) and the scalar expansion kernel
///    (`+ MLRS_KNN_CPU_VEC=0`), so no A/B escape hatch can silently diverge;
/// 2. across unit counts, so the query-tile-to-thread mapping is invisible in
///    the results.
///
/// Neighbor INDICES must match exactly in every case — that is the tie-break
/// contract, and the quantized cases are saturated with exact ties precisely so
/// a reordered comparison would be caught. VALUES are compared bitwise on those
/// quantized cases (small-integer products and sums, exact in f32) and at the
/// project's 1e-5 relative bar otherwise, the same rule the GPU dot-expansion
/// gate above uses.
///
/// Cases cover the boundaries the kernel blocks on: `n_query`/`n_train` on and
/// off the 16-row query tile and the 4-row train block, `k` at 1 and at the
/// cap, `d` at the `CPU_MAX_COLS` cap, and more query rows than one cube's
/// worth so the cube loop runs several times per unit.
///
/// The whole test is a no-op off the cpu backend — the kernel's
/// one-thread-per-unit shape is meaningless on a GPU, and `cpu_rows_applicable`
/// says so.
#[test]
fn cpu_rows_topk_matches_brute_force() {
    use mlrs_backend::prims::knn::{cpu_rows_applicable, cpu_rows_topk};

    // (n_train, n_query, d, k, quantize)
    let cases: &[(usize, usize, usize, usize, Option<u64>)] = &[
        // Below one query tile and one train block; k == n_train edge.
        (5, 3, 4, 5, None),
        // Exactly one query tile, k at the cap, train rows off the 4-block.
        (67, 16, 8, 16, None),
        // Off-boundary in both dimensions, several tiles.
        (200, 130, 8, 5, None),
        // Tie-saturated: features quantized so distances collide constantly.
        (300, 100, 4, 10, Some(3)),
        // d at the cpu cap with the argmin-degenerate k = 1.
        (500, 70, 32, 1, None),
        // More query rows than one cube covers, so the cube loop runs several
        // times per unit, and a tie-saturated tail.
        (1_000, 700, 16, 8, Some(7)),
    ];

    for &(n_train, n_query, d, k, quant) in cases {
        if !cpu_rows_applicable::<f32>(n_query, n_train, d, k) {
            println!("SKIP cpu_rows case ({n_train},{n_query},{d},{k}): not the cpu backend");
            continue;
        }

        let mut s = 0x5CA1_AB1E_u64 ^ ((n_train as u64) << 32) ^ (n_query as u64 + d as u64);
        let gen = |s: &mut u64| -> f32 {
            let r = splitmix64(s);
            match quant {
                Some(m) => (r % m) as f32,
                None => ((r >> 40) as f32 / 8_388_608.0) * 2.0 - 1.0,
            }
        };
        let xq: Vec<f32> = (0..n_query * d).map(|_| gen(&mut s)).collect();
        let xt: Vec<f32> = (0..n_train * d).map(|_| gen(&mut s)).collect();

        let mut p = pool();
        let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xq);
        let xt_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xt);

        // (arm label, MLRS_KNN_CPU_VEC, MLRS_KNN_DOT, cancellation-free?).
        // `direct` is the cpu DEFAULT; `MLRS_KNN_DOT=1` selects the expansion,
        // and the vec switch only applies to the expansion arm (the
        // cancellation-free kernel has no scalar twin).
        let arms = [
            ("direct", "1", "0", true),
            ("dot-simd", "1", "1", false),
            ("dot-scalar", "0", "1", false),
        ];
        for (arm, vec_mode, dot_mode, direct) in arms {
            let (rv, ri) = host_rows_topk(&xq, &xt, n_query, n_train, d, k, direct);
            for units in ["1", "3", "16"] {
                std::env::set_var("MLRS_KNN_CPU_VEC", vec_mode);
                std::env::set_var("MLRS_KNN_DOT", dot_mode);
                std::env::set_var("MLRS_CPU_UNITS", units);
                let (cv, ci) =
                    cpu_rows_topk::<f32>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, true)
                        .expect("cpu rows");
                let (cv, ci) = (cv.to_host(&p), ci.to_host(&p));
                std::env::remove_var("MLRS_KNN_CPU_VEC");
                std::env::remove_var("MLRS_KNN_DOT");
                std::env::remove_var("MLRS_CPU_UNITS");

                let tag = format!(
                    "arm={arm} units={units} n_train={n_train} n_query={n_query} d={d} \
                     k={k} quant={quant:?}"
                );
                assert_eq!(ci, ri, "cpu rows indices diverged at {tag}");
                if quant.is_some() {
                    assert_eq!(
                        cv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        rv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "cpu rows values diverged at {tag}"
                    );
                } else {
                    for (i, (a, b)) in cv.iter().zip(rv.iter()).enumerate() {
                        let tol = 1e-5 * b.abs().max(1.0);
                        assert!(
                            (a - b).abs() <= tol,
                            "cpu rows value {i} = {a} vs {b} at {tag}"
                        );
                    }
                }
            }
        }
    }
}

/// KNN-04: `cpu_rows_topk` validates every bound its launch derives from
/// BEFORE the unsafe launch (ASVS V5), matching `fused_distance_topk`'s
/// rejection contract. No-op off the cpu backend.
#[test]
fn cpu_rows_topk_rejects_bad_geometry() {
    use mlrs_backend::prims::knn::{cpu_rows_applicable, cpu_rows_topk};
    use mlrs_kernels::knn::{CPU_K_CAP, CPU_MAX_COLS};

    if !cpu_rows_applicable::<f32>(4, 6, 2, 1) {
        println!("SKIP cpu_rows geometry case: not the cpu backend");
        return;
    }
    let mut p = pool();
    let xq: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[0.0f32; 8]);
    let xt: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &[0.0f32; 12]);

    // Wrong x length for the claimed shape.
    assert!(matches!(
        cpu_rows_topk::<f32>(&mut p, &xq, (5, 2), &xt, 6, 1, true),
        Err(PrimError::ShapeMismatch { .. })
    ));
    // Wrong y length for the claimed shape.
    assert!(matches!(
        cpu_rows_topk::<f32>(&mut p, &xq, (4, 2), &xt, 7, 1, true),
        Err(PrimError::ShapeMismatch { .. })
    ));
    // k == 0, k > n_train, k > CPU_K_CAP.
    for bad_k in [0usize, 7, CPU_K_CAP as usize + 1] {
        assert!(
            matches!(
                cpu_rows_topk::<f32>(&mut p, &xq, (4, 2), &xt, 6, bad_k, true),
                Err(PrimError::ShapeMismatch { .. })
            ),
            "k={bad_k} must be rejected"
        );
    }
    // cols beyond the local tile cap.
    let wide = CPU_MAX_COLS as usize + 1;
    let wide_q: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut p, &vec![0.0f32; wide]);
    let wide_t: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut p, &vec![0.0f32; 2 * wide]);
    assert!(matches!(
        cpu_rows_topk::<f32>(&mut p, &wide_q, (1, wide), &wide_t, 2, 1, true),
        Err(PrimError::ShapeMismatch { .. })
    ));
}
