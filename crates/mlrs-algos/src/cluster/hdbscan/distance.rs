//! Shared host-side pairwise distance for the HDBSCAN back-end (IN-03).
//!
//! Previously duplicated verbatim across `hdbscan.rs` (the Variant-B FAST-metric
//! closure) and `centers.rs` (the medoid pairwise distance, which additionally
//! handles `Cosine`). The euclidean/manhattan/chebyshev/minkowski arms were
//! byte-identical, so a metric fix had to be applied in two places or they would
//! drift. This single implementation covers all five feature-space metrics; the
//! Variant-B callers simply never pass `Cosine` (cosine routes through the dense
//! Variant-A path), so the extra arm is harmless to them.

use super::Metric;

/// How many features a screened accumulator sums before it re-tests the
/// early-exit bound (HDBS-PERF-CPU).
///
/// The bound test is what lets a far-away point cost two features instead of
/// `p`, but a test after EVERY feature is a branch per flop, and at the `p` that
/// actually shows up (8-64) that branch costs more than the exit saves.
/// MEASURED at `n = 10_000, d = 16` on the cpu backend: the per-feature form ran
/// the Variant-B Prim in 0.854 s against 0.654 s for no screening at all — the
/// "optimization" was a 30% REGRESSION. Screening once per 8-feature block keeps
/// the exit for the high-`p` case (where it is worth real time) and costs two
/// tests per pair at `d = 16`.
///
/// The block is a screening granularity ONLY: the sum itself still runs feature
/// by feature in the original order, so every accumulator below is bit-identical
/// to [`host_pairwise`]. Re-associating the sum into per-block partials would
/// vectorize better but would also move an MST edge on a tie — the single thing
/// the `tie_break_exact` gate exists to catch.
pub(super) const SCREEN_BLOCK: usize = 8;

/// Σ(Δ²) — the pre-`sqrt` Euclidean aggregate — returning `+inf` as soon as a
/// block boundary finds the running sum at or past `bound`.
///
/// The caller passes a `bound` already mapped into the SQUARED domain (and
/// widened by a few epsilons, since `sqrt` is not exact at the boundary), so a
/// bail-out means the true distance cannot matter to it.
#[inline]
pub(super) fn sq_euclidean_screened(a: &[f64], b: &[f64], bound: f64) -> f64 {
    let n = a.len();
    let mut s = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let end = (k + SCREEN_BLOCK).min(n);
        while k < end {
            let diff = a[k] - b[k];
            s += diff * diff;
            k += 1;
        }
        if s >= bound {
            return f64::INFINITY;
        }
    }
    s
}

/// Σ|Δ| (Manhattan) with the same block screening. `fin` is the identity here,
/// so the bound needs no epsilon widening.
#[inline]
pub(super) fn manhattan_screened(a: &[f64], b: &[f64], bound: f64) -> f64 {
    let n = a.len();
    let mut s = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let end = (k + SCREEN_BLOCK).min(n);
        while k < end {
            s += (a[k] - b[k]).abs();
            k += 1;
        }
        if s >= bound {
            return f64::INFINITY;
        }
    }
    s
}

/// max|Δ| (Chebyshev) with the same block screening.
#[inline]
pub(super) fn chebyshev_screened(a: &[f64], b: &[f64], bound: f64) -> f64 {
    let n = a.len();
    let mut m = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let end = (k + SCREEN_BLOCK).min(n);
        while k < end {
            let diff = (a[k] - b[k]).abs();
            if diff > m {
                m = diff;
            }
            k += 1;
        }
        if m >= bound {
            return f64::INFINITY;
        }
    }
    m
}

/// Σ|Δ|^`pp` (Minkowski) with the same block screening. `powf` dominates the
/// inner loop, so the screen pays for itself here at any `p`.
#[inline]
pub(super) fn minkowski_screened(a: &[f64], b: &[f64], bound: f64, pp: f64) -> f64 {
    let n = a.len();
    let mut s = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let end = (k + SCREEN_BLOCK).min(n);
        while k < end {
            s += (a[k] - b[k]).abs().powf(pp);
            k += 1;
        }
        if s >= bound {
            return f64::INFINITY;
        }
    }
    s
}

/// Raw (unscaled) pairwise distance `d(i, j)` between rows `i` and `j` of the
/// row-major `n×p` host matrix `x`, under `metric`. Mirrors
/// `sklearn.metrics.pairwise_distances` for the five feature-space metrics. All
/// math is `f64`.
///
/// `Precomputed` never reaches this function: the `store_centers`-on-precomputed
/// guard in `fit` rejects the medoid path (T-15-06-V5), and the Variant-B Prim
/// only routes the FAST metrics here. Callers that divide by `alpha` themselves
/// (the Variant-B Prim) receive the RAW value — no scaling is applied here.
pub(super) fn host_pairwise(x: &[f64], p: usize, metric: Metric, i: usize, j: usize) -> f64 {
    let xi = &x[i * p..(i + 1) * p];
    let xj = &x[j * p..(j + 1) * p];
    match metric {
        Metric::Euclidean => {
            let mut s = 0.0f64;
            for k in 0..p {
                let diff = xi[k] - xj[k];
                s += diff * diff;
            }
            s.sqrt()
        }
        Metric::Manhattan => {
            let mut s = 0.0f64;
            for k in 0..p {
                s += (xi[k] - xj[k]).abs();
            }
            s
        }
        Metric::Chebyshev => {
            let mut m = 0.0f64;
            for k in 0..p {
                let diff = (xi[k] - xj[k]).abs();
                if diff > m {
                    m = diff;
                }
            }
            m
        }
        Metric::Minkowski { p: pp } => {
            let mut s = 0.0f64;
            for k in 0..p {
                s += (xi[k] - xj[k]).abs().powf(pp);
            }
            s.powf(1.0 / pp)
        }
        Metric::Cosine => {
            // 1 − x̂·ŷ (zero-norm rows map to all-zeros ⇒ distance 1).
            let ni = xi.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let nj = xj.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if ni > 0.0 && nj > 0.0 {
                let mut dot = 0.0f64;
                for k in 0..p {
                    dot += (xi[k] / ni) * (xj[k] / nj);
                }
                let d = 1.0 - dot;
                if d > 0.0 {
                    d
                } else {
                    0.0
                }
            } else {
                1.0
            }
        }
        Metric::Precomputed => {
            unreachable!(
                "host_pairwise is never called on Precomputed: store_centers errors on it \
                 (T-15-06-V5) and the Variant-B Prim only routes FAST metrics here"
            )
        }
    }
}
