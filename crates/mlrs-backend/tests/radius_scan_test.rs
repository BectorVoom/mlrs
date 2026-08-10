//! `radius_scan` — the two `radius_neighbors` arms against an independent
//! reference, and against each other (NEIGH-RADIUS-HOST / NEIGH-RADIUS-GPU).
//!
//! The estimator-level oracle (`nearest_neighbors_params_test`, and the Python
//! `test_oracle_nearest_neighbors_params.py`) compares mlrs against sklearn's
//! own answers on fixed fixtures. These tests pin the properties those fixtures
//! cannot reach:
//!
//! - `arms_agree_with_reference` — both the fused HOST scan
//!   ([`radius_host_scan`]) and the DEVICE count+compaction pair
//!   ([`radius_scan_device_tile`], driven through the same `metric_distance`
//!   tile the estimator builds) reproduce a full `O(n²)` f64 reference match set,
//!   for every metric, at three densities including "everything matches" and
//!   "nothing matches".
//! - `matches_are_ascending_by_index` — the ordering contract the CSR layout and
//!   sklearn's `sort_results=False` both rest on. It is the one property the
//!   device arm's segment compaction could plausibly break, and an atomic
//!   bump-allocator compaction WOULD break.
//! - `avx2_and_baseline_agree_bitwise` — the host arm's runtime-detected AVX2
//!   body against its baseline twin, byte for byte (`host_simd`'s claim: the
//!   lanes are independent accumulators, so widening the register cannot
//!   reassociate anything).
//! - the ASVS V5 geometry/`radius` rejections, which are `pub` prim contract.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use mlrs_backend::abflag;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::metric_distance;
use mlrs_backend::prims::host_simd::avx2_available;
use mlrs_backend::prims::knn_graph::Metric;
use mlrs_backend::prims::radius::{radius_scan_device_tile, RadiusMatches};
use mlrs_backend::prims::radius_host::radius_host_scan;
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

/// `rows × cols` of well-spread positive values — `knn_host_test`'s design, for
/// the same reasons (positive and off the origin so cosine is well conditioned;
/// spread so distances are distinct and no candidate sits ON the threshold,
/// where a membership decision would turn on the last bit of a rounding).
fn design(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|_| ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) as f32 * 4.0 + 3.0)
        .collect()
}

/// Every metric, with the Minkowski exponent split across its two lane loops:
/// `p = 3` takes the repeated-multiplication path, `p = 2.5` the `powf` one.
fn metrics() -> Vec<(&'static str, Metric)> {
    vec![
        ("euclidean", Metric::Euclidean),
        ("manhattan", Metric::Manhattan),
        ("chebyshev", Metric::Chebyshev),
        ("cosine", Metric::Cosine),
        ("minkowski3", Metric::Minkowski { p: 3.0 }),
        ("minkowski2.5", Metric::Minkowski { p: 2.5 }),
    ]
}

/// One pair's TRUE distance in f64 over the plain feature loop — deliberately
/// sharing no structure with either arm's arithmetic.
fn reference_distance(a: &[f32], b: &[f32], metric: Metric) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|&v| v as f64).collect(),
        b.iter().map(|&v| v as f64).collect(),
    );
    match metric {
        Metric::Euclidean => x
            .iter()
            .zip(&y)
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt(),
        Metric::Manhattan => x.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum(),
        Metric::Chebyshev => x
            .iter()
            .zip(&y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max),
        Metric::Minkowski { p } => x
            .iter()
            .zip(&y)
            .map(|(u, v)| (u - v).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p),
        Metric::Cosine => {
            let dot: f64 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
            let nx: f64 = x.iter().map(|v| v * v).sum();
            let ny: f64 = y.iter().map(|v| v * v).sum();
            (1.0 - dot / (nx * ny).sqrt()).clamp(0.0, 2.0)
        }
    }
}

/// The reference match set: every `(row, train)` pair whose f64 distance is
/// within `radius`, in ascending training-index order, plus the per-row counts.
fn reference_radius(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    radius: f64,
    metric: Metric,
) -> (Vec<f64>, Vec<i32>, Vec<u32>) {
    let (mut dist, mut idx, mut counts) = (Vec::new(), Vec::new(), Vec::new());
    for q in 0..n_query {
        let mut c = 0u32;
        for t in 0..n_train {
            let v = reference_distance(&xq[q * d..(q + 1) * d], &xt[t * d..(t + 1) * d], metric);
            if v <= radius {
                dist.push(v);
                idx.push(t as i32);
                c += 1;
            }
        }
        counts.push(c);
    }
    (dist, idx, counts)
}

/// Every pairwise reference distance of the design, ascending.
fn all_distances(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    metric: Metric,
) -> Vec<f64> {
    let mut all: Vec<f64> = Vec::with_capacity(n_query * n_train);
    for q in 0..n_query {
        for t in 0..n_train {
            all.push(reference_distance(
                &xq[q * d..(q + 1) * d],
                &xt[t * d..(t + 1) * d],
                metric,
            ));
        }
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all
}

/// A radius that puts roughly `frac` of the pairs inside it, and that no
/// candidate sits ON.
///
/// A quantile of the distance distribution IS one of the candidate distances,
/// so using it directly makes membership at the boundary turn on whether the
/// arm's f32 arithmetic rounded that one pair up or down — a real ambiguity, but
/// one about float rounding rather than about the scan, and it made the first
/// version of `arms_agree_with_reference` fail on a single cosine pair. This
/// walks forward to the first pair with a >=0.1% relative gap to its successor
/// (~1000x any f32 error at this width) and returns the midpoint of that gap.
fn radius_at(all: &[f64], frac: f64) -> f64 {
    let start = ((all.len() as f64 * frac) as usize).min(all.len() - 2);
    for i in start..all.len() - 1 {
        if all[i + 1] > all[i] * 1.001 + 1e-9 {
            return 0.5 * (all[i] + all[i + 1]);
        }
    }
    all[all.len() - 1] * 1.5
}

/// Run the fused HOST arm.
fn run_host(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    radius: f64,
    metric: Metric,
) -> RadiusMatches<f32> {
    let mut p = pool();
    let xq_d = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, xq);
    let xt_d = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, xt);
    let out = radius_host_scan::<f32>(&mut p, &xq_d, (n_query, d), &xt_d, n_train, radius, metric)
        .expect("host radius scan");
    xq_d.release_into(&mut p);
    xt_d.release_into(&mut p);
    out
}

/// Run the DEVICE arm over the same `metric_distance` tile the estimator builds.
fn run_device(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    radius: f64,
    metric: Metric,
) -> RadiusMatches<f32> {
    let mut p = pool();
    let xq_d = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, xq);
    let xt_d = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, xt);
    let (dist, needs_sqrt) = metric_distance::<f32>(
        &mut p,
        &xq_d,
        (n_query, d),
        &xt_d,
        (n_train, d),
        metric,
        None,
    )
    .expect("metric_distance");
    let out = radius_scan_device_tile::<f32>(
        &mut p,
        &dist,
        n_query,
        n_train,
        if needs_sqrt { radius * radius } else { radius },
        needs_sqrt,
        0,
    )
    .expect("device radius scan");
    dist.release_into(&mut p);
    xq_d.release_into(&mut p);
    xt_d.release_into(&mut p);
    out
}

/// Compare one arm's match set against the reference: membership POSITIONALLY
/// (radius membership is a plain `<=` scan in a fixed column order on both
/// engines, so unlike top-k there is no tie-order slack to allow) and the kept
/// distances within an f32 band.
fn assert_matches(got: &RadiusMatches<f32>, want: &(Vec<f64>, Vec<i32>, Vec<u32>), label: &str) {
    assert_eq!(got.counts, want.2, "{label}: per-row match counts");
    assert_eq!(got.indices, want.1, "{label}: match indices");
    assert_eq!(
        got.distances.len(),
        want.0.len(),
        "{label}: match distance count"
    );
    for (i, (&g, &w)) in got.distances.iter().zip(&want.0).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!(
            (g as f64 - w).abs() <= tol,
            "{label}: match {i} distance {g} vs reference {w}"
        );
    }
}

#[test]
fn arms_agree_with_reference() {
    let (n_query, n_train, d) = (37, 200, 9);
    let xq = design(n_query, d, 11);
    let xt = design(n_train, d, 23);

    for (name, metric) in metrics() {
        // Three densities per metric, derived from the fixture's OWN distance
        // distribution rather than a fixed number: a euclidean-scaled radius
        // matches every point under cosine (whose range is [0, 2]) and would
        // make the sparse rung vacuous.
        let all = all_distances(&xq, &xt, n_query, n_train, d, metric);
        // A radius below every distance (empty result) and one above all of them
        // (everything matches) bracket the interesting rungs.
        let radii = [
            0.0,
            radius_at(&all, 0.1),
            radius_at(&all, 0.5),
            all[all.len() - 1] * 1.5,
        ];

        for radius in radii {
            let want = reference_radius(&xq, &xt, n_query, n_train, d, radius, metric);
            let label = format!("{name} @ radius {radius:.4}");

            let host = {
                let _g = abflag::force("MLRS_RADIUS_HOST", "1");
                run_host(&xq, &xt, n_query, n_train, d, radius, metric)
            };
            assert_matches(&host, &want, &format!("host {label}"));

            let device = {
                let _g = abflag::force("MLRS_RADIUS_DEVICE", "1");
                run_device(&xq, &xt, n_query, n_train, d, radius, metric)
            };
            assert_matches(&device, &want, &format!("device {label}"));
        }
    }
}

#[test]
fn matches_are_ascending_by_index() {
    // Wide enough that the device arm cuts each row into MANY segments (its
    // SEGMENT_COLS is 128) — with a single segment per row the ordering property
    // would hold trivially and prove nothing about the compaction.
    let (n_query, n_train, d) = (5, 900, 6);
    let xq = design(n_query, d, 41);
    let xt = design(n_train, d, 43);

    for (name, metric) in metrics() {
        // Per-metric, so every metric's rows actually match something — a fixed
        // radius that is generous for cosine matches NOTHING under manhattan at
        // this width, and the ordering claim would be vacuous there.
        let radius = radius_at(&all_distances(&xq, &xt, n_query, n_train, d, metric), 0.3);
        for (arm, got) in [
            ("host", {
                let _g = abflag::force("MLRS_RADIUS_HOST", "1");
                run_host(&xq, &xt, n_query, n_train, d, radius, metric)
            }),
            ("device", {
                let _g = abflag::force("MLRS_RADIUS_DEVICE", "1");
                run_device(&xq, &xt, n_query, n_train, d, radius, metric)
            }),
        ] {
            let mut at = 0usize;
            let mut any = false;
            for (r, &c) in got.counts.iter().enumerate() {
                let row = &got.indices[at..at + c as usize];
                any |= c > 1;
                assert!(
                    row.windows(2).all(|w| w[0] < w[1]),
                    "{arm}/{name}: row {r} is not ascending by training index: {row:?}"
                );
                at += c as usize;
            }
            assert_eq!(
                at,
                got.indices.len(),
                "{arm}/{name}: counts must sum to len"
            );
            assert!(
                any,
                "{arm}/{name}: no row matched more than once — the ordering claim is vacuous"
            );
        }
    }
}

#[test]
fn avx2_and_baseline_agree_bitwise() {
    let (n_query, n_train, d) = (33, 150, 12);
    let xq = design(n_query, d, 7);
    let xt = design(n_train, d, 99);

    // The knob must actually MOVE the dispatch, or the two arms below are the
    // same body and this test proves nothing (`knn_host_test`'s vacuity guard).
    let forced_off = {
        let _g = abflag::force("MLRS_HOST_AVX2", "0");
        avx2_available()
    };
    assert!(
        !forced_off,
        "MLRS_HOST_AVX2=0 did not disable the AVX2 body"
    );
    let default_on = {
        let _g = abflag::clear("MLRS_HOST_AVX2");
        avx2_available()
    };
    if !default_on {
        eprintln!("skipping: this CPU reports no AVX2/FMA, so there is one body, not two");
        return;
    }

    for (name, metric) in metrics() {
        let radius = radius_at(&all_distances(&xq, &xt, n_query, n_train, d, metric), 0.3);
        let _h = abflag::force("MLRS_RADIUS_HOST", "1");
        let wide = {
            let _g = abflag::clear("MLRS_HOST_AVX2");
            run_host(&xq, &xt, n_query, n_train, d, radius, metric)
        };
        let base = {
            let _g = abflag::force("MLRS_HOST_AVX2", "0");
            run_host(&xq, &xt, n_query, n_train, d, radius, metric)
        };
        assert!(
            !wide.indices.is_empty(),
            "{name}: no match at all — the comparison would be vacuous"
        );
        assert_eq!(
            wide.distances
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            base.distances
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{name}: distances differ between the AVX2 and baseline bodies"
        );
        assert_eq!(wide.indices, base.indices, "{name}: indices differ");
        assert_eq!(wide.counts, base.counts, "{name}: counts differ");
    }
}

#[test]
fn host_arm_rejects_bad_geometry() {
    let mut p = pool();
    let xq = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, &design(4, 3, 1));
    let xt = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, &design(10, 3, 2));

    // A query length that does not match its declared geometry.
    let e = radius_host_scan::<f32>(&mut p, &xq, (5, 3), &xt, 10, 1.0, Metric::Euclidean)
        .expect_err("mismatched query geometry must be rejected");
    assert!(
        matches!(e, PrimError::ShapeMismatch { operand: "x", .. }),
        "{e:?}"
    );

    // A negative radius: the prim-level backstop under the estimator's own check.
    let e = radius_host_scan::<f32>(&mut p, &xq, (4, 3), &xt, 10, -0.5, Metric::Euclidean)
        .expect_err("a negative radius must be rejected");
    assert!(
        matches!(
            e,
            PrimError::ShapeMismatch {
                operand: "radius",
                ..
            }
        ),
        "{e:?}"
    );

    xq.release_into(&mut p);
    xt.release_into(&mut p);
}

#[test]
fn device_arm_rejects_bad_geometry() {
    let mut p = pool();
    let dist = DeviceArray::<ActiveRuntime, f32>::from_host(&mut p, &design(4, 10, 5));
    let e = radius_scan_device_tile::<f32>(&mut p, &dist, 4, 11, 1.0, false, 0)
        .expect_err("a tile length that is not rows*cols must be rejected");
    assert!(
        matches!(
            e,
            PrimError::ShapeMismatch {
                operand: "dist",
                ..
            }
        ),
        "{e:?}"
    );
    dist.release_into(&mut p);
}
