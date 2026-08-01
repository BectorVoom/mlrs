//! Agreement gate for the cpu-backend host k-NN scan (UMAP-PERF-CPU).
//!
//! [`umap_host_knn::host_knn`](mlrs_algos::manifold::umap_host_knn::host_knn)
//! replaces the device
//! [`knn_graph`](mlrs_backend::prims::knn_graph::knn_graph) prim inside
//! `Umap::fit` on the cpu backend for wall-clock reasons only — it must return
//! the SAME graph. These tests hold it to that: identical neighbour indices and
//! identical distances (to `f64` rounding) for all five metrics, on geometry
//! chosen to be maximally hostile to a selection tie-break.
//!
//! The geometry is a COARSE INTEGER LATTICE with a DUPLICATED row, the same
//! choice `hdbscan_test::core_distances_host_matches_device` makes: on integers
//! many pairwise distances are exactly equal, and the duplicate guarantees a
//! zero-distance pair, so a wrong tie-break (the mlrs convention is
//! lowest-index-first) shows up as an index mismatch immediately rather than as
//! a rare flake on random data.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use mlrs_algos::manifold::umap::Metric;
use mlrs_algos::manifold::umap_host_knn::host_knn;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn_graph::{knn_graph, Metric as KnnMetric};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Standard f64 capability gate (the `umap_test.rs` convention).
fn gate_f64(case: &str) -> bool {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("umap_host_knn {case} f64 backend={backend}: SKIPPED (no f64 support)");
        return true;
    }
    false
}

/// A coarse integer lattice with a duplicated row — deliberately tie-rich.
fn lattice(n: usize, d: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            // Small integer coordinates => many exactly-equal pairwise distances.
            x[i * d + j] = (((i * 7 + j * 13) % 5) as f64) - 2.0;
        }
    }
    // Duplicate row 0 into the last row: guarantees an exact zero-distance pair
    // that both engines must break by index.
    if n >= 2 {
        let (head, tail) = x.split_at_mut((n - 1) * d);
        tail[..d].copy_from_slice(&head[..d]);
    }
    x
}

fn map_metric(m: Metric) -> KnnMetric {
    match m {
        Metric::Euclidean => KnnMetric::Euclidean,
        Metric::Manhattan => KnnMetric::Manhattan,
        Metric::Cosine => KnnMetric::Cosine,
        Metric::Chebyshev => KnnMetric::Chebyshev,
        Metric::Minkowski { p } => KnnMetric::Minkowski { p },
    }
}

fn run_agreement(tag: &str, metric: Metric) {
    if gate_f64(tag) {
        return;
    }
    let (n, d, k) = (60usize, 4usize, 7usize);
    let x_host = lattice(n, d);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let knn_metric = map_metric(metric);
    let p = match knn_metric {
        KnnMetric::Minkowski { p } => p,
        _ => 2.0,
    };
    let (idx_dev, dist_dev) =
        knn_graph::<f64>(&mut pool, &x, (n, d), k, knn_metric, false, p).expect("device knn_graph");
    let dev_idx: Vec<f64> = idx_dev.to_host(&pool).iter().map(|&v| v as f64).collect();
    let dev_dist: Vec<f64> = dist_dev.to_host(&pool);
    idx_dev.release_into(&mut pool);
    dist_dev.release_into(&mut pool);

    let (h_idx, h_dist) = host_knn(&x_host, n, d, k, metric);

    assert_eq!(h_idx.len(), dev_idx.len(), "{tag}: index buffer length");
    assert_eq!(h_dist.len(), dev_dist.len(), "{tag}: distance buffer length");

    for r in 0..n {
        for c in 0..k {
            let e = r * k + c;
            // Distances first: an index mismatch on a TIE is benign only if the
            // distances still agree, so report the numeric disagreement (the real
            // defect) before the index one.
            let (hd, dd) = (h_dist[e], dev_dist[e]);
            let tol = 1e-9 * dd.abs().max(1.0);
            assert!(
                (hd - dd).abs() <= tol,
                "{tag}: row {r} slot {c}: host distance {hd} != device {dd}"
            );
            // Indices must agree EXACTLY wherever the distance is not tied with
            // its neighbour slot — under a tie either engine may legally emit
            // either index, so compare indices only when the slot is unambiguous.
            let tied_prev = c > 0 && (dev_dist[e - 1] - dd).abs() <= tol;
            let tied_next = c + 1 < k && (dev_dist[e + 1] - dd).abs() <= tol;
            if !tied_prev && !tied_next {
                assert_eq!(
                    h_idx[e] as usize, dev_idx[e] as usize,
                    "{tag}: row {r} slot {c}: host index != device index at an untied distance"
                );
            }
        }
        // Self must never appear (UMAP's include_self = false).
        for c in 0..k {
            assert_ne!(
                h_idx[r * k + c] as usize,
                r,
                "{tag}: row {r} slot {c}: host scan emitted the query itself"
            );
        }
        // The row must be ascending in distance.
        for c in 1..k {
            assert!(
                h_dist[r * k + c] >= h_dist[r * k + c - 1],
                "{tag}: row {r} not ascending at slot {c}"
            );
        }
    }
}

#[test]
fn host_knn_matches_device_euclidean() {
    run_agreement("euclidean", Metric::Euclidean);
}

#[test]
fn host_knn_matches_device_manhattan() {
    run_agreement("manhattan", Metric::Manhattan);
}

#[test]
fn host_knn_matches_device_chebyshev() {
    run_agreement("chebyshev", Metric::Chebyshev);
}

#[test]
fn host_knn_matches_device_minkowski() {
    // The minkowski kernel evaluates `F::powf`; a backend without f64
    // transcendentals cannot compile it (see
    // `capability::f64_transcendental_supported`). The other metrics are pure
    // arithmetic and stay covered.
    if capability::skip_f64_transcendental_with_log() {
        return;
    }
    run_agreement("minkowski", Metric::Minkowski { p: 3.0 });
}

#[test]
fn host_knn_matches_device_cosine() {
    run_agreement("cosine", Metric::Cosine);
}

/// The tie-break convention itself, isolated from the metric plumbing: with
/// several points at EXACTLY the same distance from the query, the host scan
/// must select them in ASCENDING INDEX order (the mlrs `top_k` convention).
#[test]
fn host_knn_breaks_ties_by_lowest_index() {
    // Row 0 at the origin; rows 1..=6 all at distance 1 along alternating axes.
    let d = 2usize;
    let x: Vec<f64> = vec![
        0.0, 0.0, // 0: query
        1.0, 0.0, // 1
        -1.0, 0.0, // 2
        0.0, 1.0, // 3
        0.0, -1.0, // 4
        1.0, 0.0, // 5 (duplicate of 1)
        -1.0, 0.0, // 6 (duplicate of 2)
    ];
    let n = x.len() / d;
    let (idx, dist) = host_knn(&x, n, d, 6, Metric::Euclidean);
    for c in 0..6 {
        assert!(
            (dist[c] - 1.0).abs() < 1e-12,
            "all six neighbours of the origin are at distance 1, slot {c} = {}",
            dist[c]
        );
        assert_eq!(
            idx[c] as usize,
            c + 1,
            "an all-tied row must come out in ascending index order at slot {c}"
        );
    }
}
