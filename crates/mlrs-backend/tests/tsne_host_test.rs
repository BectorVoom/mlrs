//! TSNE-PARAMS — the parallel HOST t-SNE engine
//! ([`mlrs_backend::prims::tsne_host`]) gates.
//!
//! Two properties are asserted here, and both are structural rather than
//! statistical — neither needs a committed sklearn fixture, because each one is
//! an identity the engine must satisfy against ITSELF.
//!
//! 1. **Barnes-Hut at `angle = 0` IS the exact objective.** θ is the threshold
//!    in `width² / dist² < θ²`, and no cell can satisfy that at `θ = 0`, so the
//!    traversal descends to individual leaves and the negative force becomes
//!    the full `O(n²)` summation. Feed the sparse arm a `P` whose sparsity
//!    pattern is COMPLETE — every off-diagonal entry present — and the positive
//!    force is exact too. At that point the two arms are computing the same
//!    gradient by two entirely different routes (a recursive quadtree walk with
//!    a deferred-logarithm KL identity, versus a dense two-pass sweep), so the
//!    embeddings must agree. This is the strongest available check on the
//!    quadtree: a mis-built tree, a dropped child, a wrong `cumulative_size`,
//!    or a barycenter update off by one point all break it.
//!
//! 2. **The result does not depend on the worker count.** The module claims
//!    every reduction runs in point order precisely so a fit is reproducible at
//!    any thread count — unlike sklearn, whose `sum_Q` and KL are OpenMP
//!    reduction variables and therefore depend on `OMP_NUM_THREADS`. That claim
//!    is only worth making if it is gated, so it is: BIT-identical output
//!    across 1, 2, 3, 5 and 8 workers.
//!
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod
//! tests`.

use mlrs_backend::prims::tsne_host::{tsne_descent, TsneDescentConfig, TsneP};

/// Counter-based splitmix64, so the designs are reproducible without a
//  dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// A normalized, symmetric, zero-diagonal joint-probability matrix — the shape
/// both arms consume.
fn random_p(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut p = vec![0.0f64; n * n];
    let mut total = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let v = uniform01(&mut s) + 1e-3;
            p[i * n + j] = v;
            p[j * n + i] = v;
            total += 2.0 * v;
        }
    }
    for v in p.iter_mut() {
        *v /= total;
    }
    p
}

fn random_embedding(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n * d).map(|_| 1e-4 * (uniform01(&mut s) - 0.5)).collect()
}

/// The dense `P` above, re-expressed as a COMPLETE CSR — every off-diagonal
/// cell stored. This is what makes the Barnes-Hut positive force exact rather
/// than a k-nearest-neighbour approximation.
fn dense_p_as_csr(p: &[f64], n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let mut indptr = vec![0usize; n + 1];
    let mut indices = Vec::with_capacity(n * (n - 1));
    let mut data = Vec::with_capacity(n * (n - 1));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                indices.push(j as u32);
                data.push(p[i * n + j]);
            }
        }
        indptr[i + 1] = indices.len();
    }
    (indptr, indices, data)
}

fn config(n: usize, d: usize, angle: f64, threads: usize, max_iter: usize) -> TsneDescentConfig {
    TsneDescentConfig {
        n,
        d,
        dof: (d as f64 - 1.0).max(1.0),
        max_iter,
        early_exaggeration: 12.0,
        learning_rate: 200.0,
        min_grad_norm: 1e-7,
        n_iter_without_progress: 300,
        angle,
        threads,
        verbose: 0,
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Property 1 (see module docs): at `angle = 0` with a complete `P`, the
/// quadtree arm reproduces the dense arm.
///
/// ## Why ten iterations and not a thousand
/// The two arms compute the SAME gradient — measured agreement is ~1e-15
/// relative after 1, 2, 5 and 10 steps. They do not stay together for a full
/// fit, and that is not a defect: t-SNE's descent is chaotic, so the last-bit
/// difference between summing the negative force through a tree walk and
/// summing it through a dense sweep amplifies geometrically. On this design it
/// measures 1e-15 at 10 iterations, 6e-10 at 20, and O(1) by 50. A gate over
/// 300 iterations would therefore be testing the Lyapunov exponent, not the
/// quadtree; ten iterations tests the quadtree, and the embedding has already
/// travelled from 5e-5 to ~60 by then, so it is nowhere near vacuous.
#[test]
fn barnes_hut_at_angle_zero_matches_the_exact_objective() {
    let (n, d) = (60usize, 2usize);
    let p = random_p(n, 7);
    let (indptr, indices, data) = dense_p_as_csr(&p, n);
    let start = random_embedding(n, d, 11);

    const ITERS: usize = 10;

    let mut y_exact = start.clone();
    let exact = tsne_descent(
        &mut y_exact,
        TsneP::Dense(&p),
        &config(n, d, 0.5, 4, ITERS),
    );

    let mut y_bh = start.clone();
    let bh = tsne_descent(
        &mut y_bh,
        TsneP::Sparse {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        },
        &config(n, d, 0.0, 4, ITERS),
    );

    // The embedding is what the two arms are really computing; the KL is the
    // scalar summary of it.
    let scale = y_exact.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    // The descent must have gone somewhere, or the comparison is empty.
    assert!(
        max_abs_diff(&y_exact, &start) > 1.0,
        "the {ITERS}-iteration reference barely moved; the gate would be vacuous"
    );
    let diff = max_abs_diff(&y_exact, &y_bh);
    assert!(
        diff <= 1e-11 * scale,
        "angle=0 barnes_hut must reproduce the exact objective: max|Δy| = {diff:e} \
         (embedding scale {scale:e})"
    );
    assert!(
        (exact.kl_divergence - bh.kl_divergence).abs() <= 1e-10 * exact.kl_divergence.abs().max(1.0),
        "kl mismatch: exact {} vs barnes_hut {}",
        exact.kl_divergence,
        bh.kl_divergence
    );
    assert_eq!(
        exact.n_iter, bh.n_iter,
        "the two arms must take the same stopping decision"
    );

    // Guard against the gate passing vacuously: a COARSE angle on the same
    // input must visibly differ, or `angle` is not reaching the traversal at
    // all and the comparison above proves nothing.
    let mut y_coarse = start.clone();
    tsne_descent(
        &mut y_coarse,
        TsneP::Sparse {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        },
        &config(n, d, 1.0, 4, ITERS),
    );
    assert!(
        max_abs_diff(&y_exact, &y_coarse) > 1e-11 * scale,
        "angle=1 produced the exact result too — the summary test is inert"
    );
}

/// Property 2 (see module docs): BIT-identical output at any worker count, for
/// BOTH objectives. Every reduction is written per point and summed in point
/// order specifically so this holds.
#[test]
fn both_objectives_are_bit_identical_across_worker_counts() {
    let (n, d) = (80usize, 2usize);
    let p = random_p(n, 13);
    let (indptr, indices, data) = dense_p_as_csr(&p, n);
    let start = random_embedding(n, d, 17);

    for label in ["exact", "barnes_hut"] {
        let mut baseline: Option<(Vec<f64>, f64, usize)> = None;
        for threads in [1usize, 2, 3, 5, 8] {
            let mut y = start.clone();
            let cfg = config(n, d, 0.5, threads, 400);
            let out = if label == "exact" {
                tsne_descent(&mut y, TsneP::Dense(&p), &cfg)
            } else {
                tsne_descent(
                    &mut y,
                    TsneP::Sparse {
                        indptr: &indptr,
                        indices: &indices,
                        data: &data,
                    },
                    &cfg,
                )
            };
            match &baseline {
                None => baseline = Some((y, out.kl_divergence, out.n_iter)),
                Some((y0, kl0, it0)) => {
                    assert_eq!(
                        &y, y0,
                        "{label}: threads={threads} changed the embedding; a reduction \
                         is not running in point order"
                    );
                    assert_eq!(out.kl_divergence, *kl0, "{label}: threads={threads} changed the KL");
                    assert_eq!(out.n_iter, *it0, "{label}: threads={threads} changed n_iter");
                }
            }
        }
    }
}

/// The quadtree must handle DUPLICATE points. `_quad_tree.pyx` absorbs a point
/// within `EPSILON` of a leaf's resident into that leaf's `cumulative_size`
/// rather than splitting forever; without that rule a design with repeated rows
/// recurses until it exhausts memory.
#[test]
fn duplicate_points_do_not_diverge_the_tree() {
    let (n, d) = (64usize, 2usize);
    let p = random_p(n, 23);
    let (indptr, indices, data) = dense_p_as_csr(&p, n);

    // Half the rows are exact duplicates of the other half.
    let mut start = random_embedding(n / 2, d, 29);
    start.extend_from_within(..);

    let mut y = start.clone();
    let out = tsne_descent(
        &mut y,
        TsneP::Sparse {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        },
        &config(n, d, 0.5, 4, 300),
    );
    assert!(
        y.iter().all(|v| v.is_finite()),
        "a design with duplicate rows must still produce a finite embedding"
    );
    assert!(
        out.kl_divergence.is_finite(),
        "kl must be finite for a duplicate-heavy design, got {}",
        out.kl_divergence
    );
}

