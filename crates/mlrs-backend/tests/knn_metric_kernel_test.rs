//! KNN-CUBE-METRIC correctness gate: the CubeCL metric row-scan kernel family
//! (`prims::knn::cpu_metric_rows_topk`).
//!
//! These are the kernels that took the cpu k-NN search off the GPU-shaped
//! `distance → top_k` composition for every non-Euclidean metric and for
//! `k > 16` / `n_features > 32`. They are ordinary `#[cube]` kernels whose lane
//! axis is a `Vector<F, Const<32>>` of QUERY ROWS, so `cubecl-cpu` lowers each
//! per-feature step to one MLIR `vector<32xf32>` op — including `abs`, `max`,
//! `powf`, `sqrt` and the cosine `select`, all of which `Vector` implements.
//!
//! What is checked:
//!
//! - `matches_a_brute_force_reference_on_every_metric` — an independent, naive,
//!   f64 host reference per metric, compared for BOTH the `k` selection and the
//!   emitted distances. It shares no structure with the kernel: no tile
//!   transposition, no lane blocking, no deferred boundary root.
//! - `covers_the_shapes_the_euclidean_kernel_cannot` — `k` past the tuned
//!   kernel's 16-slot list and `n_features` past its 32-column cache, which is
//!   the whole reason this family exists.
//! - `ties_resolve_to_the_lowest_index` — the `(value, index)` tie-break every
//!   other arm uses, on an all-duplicates training set.
//! - `agrees_with_the_host_arm` — the family against `knn_host`, which is
//!   independently gated against the same reference. They are different
//!   implementations of one contract, so a divergence means one of them drifted.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::knn::{cpu_metric_rows_applicable, cpu_metric_rows_topk};
use mlrs_backend::prims::knn_graph::Metric;
use mlrs_backend::prims::knn_host::knn_host_topk;
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

/// Well-spread positive rows: off the origin so the cosine cases are well
/// conditioned, and spread so pairwise distances are distinct (a tie would make
/// the reference's neighbour choice ambiguous for reasons unrelated to the
/// kernel — `ties_resolve_to_the_lowest_index` covers ties deliberately).
fn design(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|_| ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64) as f32 * 4.0 + 3.0)
        .collect()
}

/// Every metric the family implements, with the Minkowski exponent at a genuine
/// non-degenerate value (`p != 1, 2`, so it exercises the `powf` lane loop
/// rather than a collapsed fast path).
fn metrics() -> Vec<(&'static str, Metric)> {
    vec![
        ("euclidean", Metric::Euclidean),
        ("manhattan", Metric::Manhattan),
        ("chebyshev", Metric::Chebyshev),
        ("cosine", Metric::Cosine),
        ("minkowski3", Metric::Minkowski { p: 3.0 }),
    ]
}

/// One pair's distance, computed the most obvious way there is — f64, plain
/// feature loop, root applied immediately.
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

/// The `k` nearest training rows of every query row, by full sort under the same
/// `(value, index)` total order the kernels admit with.
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
        row.sort_by(|a, b| a.partial_cmp(b).expect("finite reference distances"));
        for (v, i) in row.into_iter().take(k) {
            val.push(v);
            idx.push(i);
        }
    }
    (val, idx)
}

/// Launch the kernel family and read both results back.
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
        cpu_metric_rows_topk::<f32>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, metric)
            .expect("cpu_metric_rows_topk");
    let out = (val.to_host(&p), idx.to_host(&p));
    val.release_into(&mut p);
    idx.release_into(&mut p);
    out
}

/// The project's oracle band, relative to the value's own magnitude.
fn tol(want: f64) -> f64 {
    1e-5 * want.abs().max(1.0)
}

/// Assert a selection matches the reference, allowing a tie-order swap ONLY
/// where the two candidates are indistinguishable at this precision.
///
/// ## Why not positional index equality
/// Exact positional equality is the right assert for four of the five metrics
/// and this used to make it for all five. COSINE cannot satisfy it in f32: its
/// value is `1 − dot/‖x‖‖y‖`, and on the positive-orthant design here the
/// similarity sits near 0.956, so the subtraction cancels away most of the
/// mantissa. The kernel accumulates its dot over `d` features as a sequential
/// `fma` chain whose error is ~`√d · eps` relative — about `1e-6` ABSOLUTE on a
/// distance of `4.4e-2` at `d = 48`. Two training rows `6.4e-7` apart (measured:
/// row 7, slots 26/27) are therefore genuinely below the arithmetic's
/// resolution, and which one sorts first is not determined by the problem.
///
/// So the contract asserted here is the one that is actually true:
///
/// 1. the emitted DISTANCES match the reference positionally — this is what
///    catches a wrong metric, a wrong root, a wrong norm;
/// 2. where an INDEX differs, the neighbour the kernel picked must sit within
///    the same band of the reference's pick. A genuinely wrong neighbour is
///    orders of magnitude outside it; a tie is inside by construction.
#[allow(clippy::too_many_arguments)]
fn assert_topk_matches(
    name: &str,
    metric: Metric,
    xq: &[f32],
    xt: &[f32],
    d: usize,
    k: usize,
    got: &(Vec<f32>, Vec<u32>),
    want: &(Vec<f64>, Vec<u32>),
) {
    let (got_val, got_idx) = got;
    let (want_val, want_idx) = want;
    assert_eq!(got_val.len(), want_val.len(), "{name}: result length");
    for (slot, (&gv, &wv)) in got_val.iter().zip(want_val).enumerate() {
        assert!(
            (gv as f64 - wv).abs() <= tol(wv),
            "{name}: distance at slot {slot} = {gv}, want {wv}"
        );
        if got_idx[slot] != want_idx[slot] {
            let q = slot / k;
            let picked = got_idx[slot] as usize;
            let picked_d = reference_distance(
                &xq[q * d..(q + 1) * d],
                &xt[picked * d..(picked + 1) * d],
                metric,
            );
            assert!(
                (picked_d - wv).abs() <= tol(wv),
                "{name}: slot {slot} picked index {picked} at distance {picked_d}, but the \
                 reference picked {} at {wv} — too far apart to be a tie",
                want_idx[slot]
            );
        }
    }
}

/// This family is cpu-only by construction — its parallelism is `Vector` LANES,
/// where a GPU backend wants units. Everything below is a no-op elsewhere.
fn on_cpu() -> bool {
    let applicable = cpu_metric_rows_applicable::<f32>(8, 32, 4, 3);
    if !applicable {
        eprintln!("skipping: the metric row-scan family is cpu-only");
    }
    applicable
}

#[test]
fn matches_a_brute_force_reference_on_every_metric() {
    if !on_cpu() {
        return;
    }
    let (n_query, n_train, d, k) = (37, 200, 9, 7);
    let xq = design(n_query, d, 11);
    let xt = design(n_train, d, 42);

    for (name, metric) in metrics() {
        let got = run(&xq, &xt, n_query, n_train, d, k, metric);
        let want = reference_topk(&xq, &xt, n_query, n_train, d, k, metric);
        assert_topk_matches(name, metric, &xq, &xt, d, k, &got, &want);
    }
}

#[test]
fn covers_the_shapes_the_euclidean_kernel_cannot() {
    if !on_cpu() {
        return;
    }
    // `k = 40` is past the tuned kernel's 16-slot list and `d = 48` past its
    // 32-column cache — the rectangle whose only previous option was the
    // GPU-shaped composition.
    let (n_query, n_train, d, k) = (20, 120, 48, 40);
    let xq = design(n_query, d, 3);
    let xt = design(n_train, d, 5);

    for (name, metric) in metrics() {
        let got = run(&xq, &xt, n_query, n_train, d, k, metric);
        let want = reference_topk(&xq, &xt, n_query, n_train, d, k, metric);
        assert_topk_matches(name, metric, &xq, &xt, d, k, &got, &want);
    }
}

#[test]
fn ties_resolve_to_the_lowest_index() {
    if !on_cpu() {
        return;
    }
    // Every training row identical, so EVERY candidate ties and the tie-break
    // alone decides the list.
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
fn distances_are_ascending_per_row() {
    if !on_cpu() {
        return;
    }
    // Consumers rely on the ordering: `knn_regress_gather_weighted` reads column
    // 0 as the nearest neighbour, and the shim hands `kneighbors` straight out.
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
fn agrees_with_the_host_arm() {
    if !on_cpu() {
        return;
    }
    // Two independent implementations of one contract — the CubeCL kernel family
    // and `knn_host`'s plain-Rust scan. `knn_host` remains the arm past this
    // family's caps, so the two must not drift where they overlap.
    let (n_query, n_train, d, k) = (33, 150, 12, 5);
    let xq = design(n_query, d, 7);
    let xt = design(n_train, d, 99);

    for (name, metric) in metrics() {
        let kernel = run(&xq, &xt, n_query, n_train, d, k, metric);

        let mut p = pool();
        let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xq);
        let xt_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &xt);
        let (hv, hi) =
            knn_host_topk::<f32>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, metric)
                .expect("knn_host_topk");
        let host = (hv.to_host(&p), hi.to_host(&p));
        hv.release_into(&mut p);
        hi.release_into(&mut p);

        assert_eq!(kernel.1, host.1, "{name}: indices differ from the host arm");
        for (&a, &b) in kernel.0.iter().zip(&host.0) {
            // NOT bitwise: the kernel accumulates with `fma` (one rounding) where
            // the host arm uses `mul` + `add` (two), and the Minkowski root is
            // applied at different points. Both are inside the project band, and
            // the SELECTION — asserted exactly above — is what has to agree.
            assert!(
                (a as f64 - b as f64).abs() <= 1e-5 * (b as f64).abs().max(1.0),
                "{name}: {a} vs host {b}"
            );
        }
    }
}

#[test]
fn f64_matches_the_reference_too() {
    if !on_cpu() {
        return;
    }
    // The kernels are monomorphized per float width; f32 coverage does not imply
    // f64 coverage, and `Vector<f64, 32>` is four times the register pressure.
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
            cpu_metric_rows_topk::<f64>(&mut p, &xq_dev, (n_query, d), &xt_dev, n_train, k, metric)
                .expect("cpu_metric_rows_topk f64");
        let got_val = val.to_host(&p);
        let got_idx = idx.to_host(&p);
        val.release_into(&mut p);
        idx.release_into(&mut p);

        let (want_val, want_idx) = reference_topk(&xq32, &xt32, n_query, n_train, d, k, metric);
        assert_eq!(got_idx, want_idx, "{name}: f64 indices");
        for (&got, &want) in got_val.iter().zip(&want_val) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{name}: f64 distance {got} want {want}"
            );
        }
    }
}

#[test]
fn rejects_bad_geometry() {
    if !on_cpu() {
        return;
    }
    let mut p = pool();
    let xq: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &design(4, 3, 1));
    let xt: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &design(10, 3, 2));

    for (shape, n_train, k, operand) in [
        ((5usize, 3usize), 10usize, 3usize, "x"),
        ((4, 3), 11, 3, "y"),
        ((4, 3), 10, 11, "k"),
        ((4, 3), 10, 0, "k"),
    ] {
        // `DeviceArray` has no `Debug` (a device handle has nothing printable),
        // so the Ok arm is discarded rather than formatted.
        let got =
            cpu_metric_rows_topk::<f32>(&mut p, &xq, shape, &xt, n_train, k, Metric::Euclidean)
                .map(|_| ())
                .expect_err("expected a geometry rejection");
        match got {
            PrimError::ShapeMismatch { operand: op, .. } => assert_eq!(op, operand),
            other => panic!("expected a {operand} ShapeMismatch, got {other:?}"),
        }
    }
}
