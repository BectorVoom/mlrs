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
