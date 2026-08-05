//! KNN-HOST correctness gate: `prims::knn_host`, the plain-Rust worker-pool
//! `distance → top-k` scan that serves the cpu backend.
//!
//! A perf arm is only allowed to be faster, never different. The tests here pin
//! the three ways this one could differ from the device pipeline it replaced:
//!
//! - `matches_a_brute_force_reference_on_every_metric` computes the whole
//!   `n_query × n_train` distance matrix with an independent, deliberately naive
//!   host reference per metric and asserts the scan selects the same `k` — the
//!   scan's blocking, its lane transposition and its deferred boundary roots are
//!   all invisible to that reference, so any of them getting the answer wrong
//!   shows up here.
//! - `avx2_and_baseline_agree_bitwise` asserts the runtime-detected AVX2 body
//!   and the baseline one produce IDENTICAL bytes. The claim they rest on is
//!   that widening a vector cannot reassociate anything, because the lanes are
//!   independent accumulators — this is that claim as a test.
//! - `ties_resolve_to_the_lowest_index` pins the `(value, index)` tie-break the
//!   device kernels use, on a training set built entirely out of duplicates so
//!   every candidate is a tie.
//!
//! The rest pin what the tuned device kernel could NOT do (`k` past its 16-slot
//! list, `n_features` past its 32-column cache) and the ASVS V5 geometry
//! rejections.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use mlrs_backend::abflag;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn_graph::Metric;
use mlrs_backend::prims::host_simd::avx2_available;
use mlrs_backend::prims::knn_host::{knn_host_applicable, knn_host_topk};
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

/// `rows × cols` of well-spread positive values.
///
/// Positive and off the origin so the cosine cases are well conditioned (a
/// near-zero-norm row makes cosine distance numerically meaningless), and spread
/// widely so pairwise distances are distinct — a tie would make the reference's
/// neighbour choice ambiguous for reasons that have nothing to do with the scan.
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

/// One pair's distance, computed the most obvious way there is.
///
/// Deliberately NOT the scan's arithmetic: it accumulates in f64 over the plain
/// feature loop and applies each metric's root immediately, so it shares no
/// structure with the thing it is checking.
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

/// The `k` nearest training rows of every query row, by full sort.
fn reference_topk(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    k: usize,
    metric: Metric,
) -> (Vec<f64>, Vec<u32>) {
    let mut val = Vec::with_capacity(n_query * k);
    let mut idx = Vec::with_capacity(n_query * k);
    for q in 0..n_query {
        let mut row: Vec<(f64, u32)> = (0..n_train)
            .map(|t| {
                (
                    reference_distance(&xq[q * d..(q + 1) * d], &xt[t * d..(t + 1) * d], metric),
                    t as u32,
                )
            })
            .collect();
        // The same `(value, index)` total order the scan admits under, so the
        // tie-break is compared rather than dodged.
        row.sort_by(|a, b| a.partial_cmp(b).expect("finite reference distances"));
        for (v, i) in row.into_iter().take(k) {
            val.push(v);
            idx.push(i);
        }
    }
    (val, idx)
}

/// Run the scan and read both results back.
fn run(
    xq: &[f32],
    xt: &[f32],
    n_query: usize,
    n_train: usize,
    d: usize,
    k: usize,
    metric: Metric,
) -> (Vec<f32>, Vec<u32>) {
    let mut p = pool();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, xq);
    let xt_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, xt);
    let (val, idx) =
        knn_host_topk::<f32>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, metric)
            .expect("knn_host_topk");
    let out = (val.to_host(&p), idx.to_host(&p));
    val.release_into(&mut p);
    idx.release_into(&mut p);
    out
}

#[test]
fn matches_a_brute_force_reference_on_every_metric() {
    let (n_query, n_train, d, k) = (37, 200, 9, 7);
    let xq = design(n_query, d, 11);
    let xt = design(n_train, d, 42);

    for (name, metric) in metrics() {
        let (val, idx) = run(&xq, &xt, n_query, n_train, d, k, metric);
        let (want_val, want_idx) = reference_topk(&xq, &xt, n_query, n_train, d, k, metric);

        assert_eq!(idx, want_idx, "{name}: neighbour indices");
        for (j, (&got, &want)) in val.iter().zip(&want_val).enumerate() {
            // f32 accumulation against an f64 reference: a relative band, not an
            // absolute one, because the metrics differ in scale by an order of
            // magnitude (a Chebyshev distance is one feature's gap; a Manhattan
            // one is the sum of nine).
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (got as f64 - want).abs() <= tol,
                "{name}: distance {j} = {got} want {want}"
            );
        }
    }
}

#[test]
fn avx2_and_baseline_agree_bitwise() {
    let (n_query, n_train, d, k) = (33, 150, 12, 5);
    let xq = design(n_query, d, 7);
    let xt = design(n_train, d, 99);

    // The knob must actually MOVE the dispatch, or the two arms below are the
    // same body and this test proves nothing. `avx2_available` caches the CPUID
    // and environment halves of its answer, and before it was split from the
    // thread-local override half, this assertion was exactly what failed to
    // hold — the test passed while comparing the AVX2 body against itself.
    let forced_off = {
        let _g = abflag::force("MLRS_HOST_AVX2", "0");
        avx2_available()
    };
    assert!(!forced_off, "MLRS_HOST_AVX2=0 did not disable the AVX2 body");
    let default_on = {
        let _g = abflag::clear("MLRS_HOST_AVX2");
        avx2_available()
    };
    if !default_on {
        eprintln!("skipping: this CPU reports no AVX2/FMA, so there is one body, not two");
        return;
    }

    for (name, metric) in metrics() {
        // Thread-local overrides, never `set_var`: libtest runs these tests on
        // parallel threads, so mutating the process environment would race every
        // sibling's read of the same knob and could make this assertion compare
        // one body against itself.
        let wide = {
            let _g = abflag::clear("MLRS_HOST_AVX2");
            run(&xq, &xt, n_query, n_train, d, k, metric)
        };
        let base = {
            let _g = abflag::force("MLRS_HOST_AVX2", "0");
            run(&xq, &xt, n_query, n_train, d, k, metric)
        };
        assert_eq!(
            wide.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            base.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: distances differ between the AVX2 and baseline bodies"
        );
        assert_eq!(wide.1, base.1, "{name}: indices differ");
    }
}

#[test]
fn ties_resolve_to_the_lowest_index() {
    // Every training row identical, so EVERY candidate ties with every other and
    // the tie-break alone decides the whole list.
    let (n_train, d, k) = (40, 4, 6);
    let row = [1.0f32, 2.0, 3.0, 4.0];
    let xt: Vec<f32> = (0..n_train).flat_map(|_| row).collect();
    let xq = vec![9.0f32, 8.0, 7.0, 6.0];

    for (name, metric) in metrics() {
        let (_, idx) = run(&xq, &xt, 1, n_train, d, k, metric);
        assert_eq!(
            idx,
            (0..k as u32).collect::<Vec<_>>(),
            "{name}: an all-ties row must keep the k lowest indices, in order"
        );
    }
}

#[test]
fn serves_the_shapes_the_tuned_kernel_cannot() {
    // `k` past the device kernel's 16-slot list and `n_features` past its
    // 32-column local cache — the two caps whose fallback path was the whole
    // reason this arm exists.
    let (n_query, n_train, d, k) = (20, 120, 48, 40);
    let xq = design(n_query, d, 3);
    let xt = design(n_train, d, 5);

    let (val, idx) = run(&xq, &xt, n_query, n_train, d, k, Metric::Euclidean);
    let (want_val, want_idx) = reference_topk(&xq, &xt, n_query, n_train, d, k, Metric::Euclidean);
    assert_eq!(idx, want_idx);
    for (&got, &want) in val.iter().zip(&want_val) {
        assert!((got as f64 - want).abs() <= 1e-5 * want.abs().max(1.0));
    }
}

#[test]
fn distances_are_ascending_per_row() {
    // The consumers rely on this ordering: `knn_regress_gather_weighted` reads
    // column 0 as the nearest neighbour, and the Python shim's `kneighbors`
    // hands the result straight to callers who slice it.
    let (n_query, n_train, d, k) = (12, 80, 6, 9);
    let xq = design(n_query, d, 21);
    let xt = design(n_train, d, 22);
    for (name, metric) in metrics() {
        let (val, _) = run(&xq, &xt, n_query, n_train, d, k, metric);
        for q in 0..n_query {
            for j in 1..k {
                assert!(
                    val[q * k + j] >= val[q * k + j - 1],
                    "{name}: row {q} is not ascending at column {j}"
                );
            }
        }
    }
}

#[test]
fn rejects_bad_geometry() {
    let mut p = pool();
    let xq: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &design(4, 3, 1));
    let xt: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &design(10, 3, 2));

    // A query geometry that does not match the buffer length.
    assert!(matches!(
        knn_host_topk::<f32>(&mut p, &xq, (5, 3), &xt, 10, 3, Metric::Euclidean),
        Err(PrimError::ShapeMismatch { operand: "x", .. })
    ));
    // A training geometry that does not.
    assert!(matches!(
        knn_host_topk::<f32>(&mut p, &xq, (4, 3), &xt, 11, 3, Metric::Euclidean),
        Err(PrimError::ShapeMismatch { operand: "y", .. })
    ));
    // `k` past the training set, and `k = 0`.
    assert!(matches!(
        knn_host_topk::<f32>(&mut p, &xq, (4, 3), &xt, 10, 11, Metric::Euclidean),
        Err(PrimError::ShapeMismatch { operand: "k", .. })
    ));
    assert!(matches!(
        knn_host_topk::<f32>(&mut p, &xq, (4, 3), &xt, 10, 0, Metric::Euclidean),
        Err(PrimError::ShapeMismatch { operand: "k", .. })
    ));
}

#[test]
fn the_knob_forces_both_directions() {
    // The gate must be forceable ON where a tuned arm exists (that is how its
    // A/B table was measured) and OFF everywhere (that is how a KNN-HOST
    // regression is bisected). A knob that only reads the backend name would
    // make both measurements silently vacuous.
    {
        let _g = abflag::force("MLRS_KNN_HOST", "1");
        assert!(knn_host_applicable(10, 100, 4, 5, true));
    }
    {
        let _g = abflag::force("MLRS_KNN_HOST", "0");
        assert!(!knn_host_applicable(10, 100, 4, 5, false));
    }
}

#[test]
fn f64_matches_the_reference_too() {
    // The scan is monomorphized per host float; f32 and f64 share no compiled
    // code, so covering only f32 would leave half of it unexercised.
    let (n_query, n_train, d, k) = (15, 90, 5, 4);
    let xq32 = design(n_query, d, 31);
    let xt32 = design(n_train, d, 32);
    let xq: Vec<f64> = xq32.iter().map(|&v| v as f64).collect();
    let xt: Vec<f64> = xt32.iter().map(|&v| v as f64).collect();

    let mut p = pool();
    let xq_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &xq);
    let xt_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &xt);

    for (name, metric) in metrics() {
        let (val, idx) =
            knn_host_topk::<f64>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, metric)
                .expect("knn_host_topk f64");
        let got_val = val.to_host(&p);
        let got_idx = idx.to_host(&p);
        val.release_into(&mut p);
        idx.release_into(&mut p);

        let (want_val, want_idx) = reference_topk(&xq32, &xt32, n_query, n_train, d, k, metric);
        assert_eq!(got_idx, want_idx, "{name}: f64 indices");
        for (&got, &want) in got_val.iter().zip(&want_val) {
            // Both sides are f64 here, but the operands were narrowed from f32,
            // so the band stays the f32-input one rather than pretending to a
            // precision the data does not carry.
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{name}: f64 distance {got} want {want}"
            );
        }
    }
}
