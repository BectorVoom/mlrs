//! `mutual_info_classif` / `mutual_info_regression` (FSEL-01) — the two
//! k-nearest-neighbour mutual-information estimators in
//! `sklearn.feature_selection`.
//!
//! Unlike the closed-form scores in [`super::score`], these are nonparametric
//! entropy estimators:
//!
//! * continuous-vs-continuous → Kraskov, Stögbauer & Grassberger (Phys. Rev. E
//!   69, 2004), as sklearn's `_compute_mi_cc`;
//! * continuous-vs-discrete → Ross (PLoS ONE 9(2), 2014), as sklearn's
//!   `_compute_mi_cd`;
//! * discrete-vs-discrete → the plug-in contingency-table estimator,
//!   `sklearn.metrics.mutual_info_score`.
//!
//! Both k-NN estimators are `max(0, ·)`-clamped, because a negative estimate of
//! a provably non-negative quantity means "close to zero" rather than
//! "negative".
//!
//! ## Host, and why that is not a concession
//! These are neighbour-count algorithms over ONE or TWO dimensions at a time
//! (`_compute_mi_cc` searches in the 2-D `(x, y)` plane, `_compute_mi_cd` in
//! 1-D), driven by `digamma` — a host scalar special function this workspace
//! already owns. There is no `O(n·d²)` dense-linear-algebra core for a device to
//! accelerate, the searches are branch-and-sort shaped, and the estimator is
//! `O(d)` INDEPENDENT single-column problems, which is the shape the host worker
//! pool is for. Same reasoning as `ARIMA`'s Kalman pass and `HDBSCAN`'s host
//! core-distance scan.
//!
//! ## Exactness of the neighbour searches
//! sklearn uses `NearestNeighbors`/`KDTree`, which are EXACT structures — they
//! accelerate the search without approximating it. So a brute-force or
//! sort-based search returns the identical counts, and this module uses:
//!
//! * a SORT + binary search for every 1-D radius count (`nx`, `ny`, `m_all`),
//!   which is `O(n log n)` and exactly what a 1-D KD-tree computes;
//! * a brute-force scan for the 2-D Chebyshev k-th-neighbour distance in
//!   `_compute_mi_cc`, `O(n²)` per feature.
//!
//! The `O(n²)` scan is the identified perf lever, and it is deliberately not
//! pre-optimised: correctness against sklearn comes first, and a 2-D
//! Chebyshev-metric KD-tree (the repo already has a Euclidean one in
//! `cluster::hdbscan::kdtree`) is a self-contained follow-up that cannot change
//! any value.
//!
//! ## "Exact" is about the SET of neighbours, not about the last bits
//! Exact structures agree on WHICH points are nearest; they do not agree on the
//! floating-point distance they report. `NearestNeighbors(algorithm='auto')`
//! resolves to a KD-tree or to brute force depending on `n_neighbors` versus
//! `n_samples`, and the two evaluate the Euclidean distance by different
//! formulas — `sqrt((a − b)²)` against the GEMM identity `a² − 2ab + b²`. They
//! differ by a few ULP, and this estimator counts neighbours inside a radius set
//! ONE ULP below that distance, so a few ULP is worth whole integer counts.
//! [`knn_1d_kth`] therefore reproduces sklearn's dispatch rather than picking
//! whichever search is convenient. Only `_compute_mi_cd` is affected:
//! `_compute_mi_cc` searches in the CHEBYSHEV metric, which has no squared
//! reduced form, so both of its paths return the exact `max|a − b|`.
//!
//! ## The `nextafter(r, 0)` detail is load-bearing
//! Both estimators shrink the k-th neighbour distance by one ULP toward zero
//! before counting inside it (`np.nextafter(radius, 0)`), so the k-th neighbour
//! ITSELF falls outside the count. Without it every count is one too high and
//! the whole estimate shifts by `ψ(k+1) − ψ(k)`. Reproduced with
//! [`next_toward_zero`].
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::collections::HashMap;

use mlrs_backend::prims::special::digamma;

use crate::error::AlgoError;

use super::numpy_rng::{
    numpy_mean, numpy_mean_axis0, numpy_std, numpy_std_axis0, NumpyRandomState,
};

/// Which columns of `X` are DISCRETE — sklearn's `discrete_features`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscreteFeatures {
    /// sklearn's `"auto"`: `False` for a dense `X` (the only kind mlrs ingests),
    /// so every column is treated as continuous. Kept as its own variant rather
    /// than collapsed to `All(false)` so `get_params()` round-trips the string
    /// the user passed, which sklearn's `clone` contract requires.
    Auto,
    /// sklearn's `bool`: all columns discrete (`true`) or all continuous.
    All(bool),
    /// sklearn's boolean-mask / index-array form, normalised to a mask.
    Mask(Vec<bool>),
}

impl Default for DiscreteFeatures {
    fn default() -> Self {
        Self::Auto
    }
}

impl DiscreteFeatures {
    /// The sklearn spelling of the non-mask arms: `'auto'`, `'true'` or
    /// `'false'`.
    ///
    /// [`DiscreteFeatures::Mask`] renders as `"mask"` and its payload rides in a
    /// companion `BOOL` tensor — the same split
    /// [`KMeansInit`](crate::cluster::kmeans::KMeansInit) makes for an explicit
    /// init array, and for the same reason: an array is not a scalar and putting
    /// it in `__metadata__` would mean encoding a vector as text.
    pub fn name(&self) -> &'static str {
        match self {
            DiscreteFeatures::Auto => "auto",
            DiscreteFeatures::All(true) => "true",
            DiscreteFeatures::All(false) => "false",
            DiscreteFeatures::Mask(_) => "mask",
        }
    }

    /// The inverse of [`DiscreteFeatures::name`] for the three scalar arms;
    /// `None` for an unrecognised string AND for `"mask"`, whose payload the
    /// caller must supply from the companion tensor.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(DiscreteFeatures::Auto),
            "true" => Some(DiscreteFeatures::All(true)),
            "false" => Some(DiscreteFeatures::All(false)),
            _ => None,
        }
    }

    /// Resolve to a length-`d` mask, `AlgoError` if an explicit mask has the
    /// wrong length.
    fn resolve(&self, d: usize) -> Result<Vec<bool>, AlgoError> {
        match self {
            // `"auto"` is `issparse(X)`, and mlrs ingests dense Arrow only
            // (base.py `__sklearn_tags__` turns the sparse tag off), so `auto`
            // is unconditionally "all continuous" here. Stated explicitly
            // because it is the one place mlrs's narrower input domain changes
            // what a sklearn default MEANS rather than merely what it accepts.
            Self::Auto => Ok(vec![false; d]),
            Self::All(v) => Ok(vec![*v; d]),
            Self::Mask(m) => {
                if m.len() != d {
                    return Err(AlgoError::Prim(mlrs_core::PrimError::ShapeMismatch {
                        operand: "discrete_features",
                        rows: 1,
                        cols: d,
                        len: m.len(),
                    }));
                }
                Ok(m.clone())
            }
        }
    }
}

/// The full parameter surface both `mutual_info_*` functions share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutualInfoParams {
    /// sklearn `discrete_features='auto'`.
    pub discrete_features: DiscreteFeatures,
    /// sklearn `n_neighbors=3`.
    pub n_neighbors: usize,
    /// sklearn `copy=True`. Retained for API parity and INERT in Rust: the
    /// estimator always works on its own `f64` buffer (the ingress widened into
    /// one), so there is no caller array to overwrite and no copy to skip.
    /// Kept so `get_params()` round-trips it and a ported call site compiles.
    pub copy: bool,
    /// sklearn `random_state=None`. `Some(seed)` reproduces
    /// `numpy.random.RandomState(seed)` bit-for-bit (see
    /// [`super::numpy_rng`]).
    ///
    /// `None` seeds `0` here, where sklearn draws from numpy's process-global
    /// stream. A DELIBERATE divergence: the noise is a tie-breaker of magnitude
    /// `1e-10`, and a reproducible score is worth more than reproducing
    /// global-state behaviour that sklearn's own docs describe as
    /// non-reproducible. Pass an explicit seed when comparing against sklearn.
    pub random_state: Option<u64>,
    /// sklearn `n_jobs=None`. `None`/`Some(1)` runs the per-column loop inline;
    /// `Some(k)` splits the COLUMNS across `k` threads (sklearn parallelises the
    /// same axis); `Some(0)` is rejected by the caller as sklearn does.
    pub n_jobs: Option<usize>,
}

impl Default for MutualInfoParams {
    fn default() -> Self {
        Self {
            discrete_features: DiscreteFeatures::Auto,
            n_neighbors: 3,
            copy: true,
            random_state: None,
            n_jobs: None,
        }
    }
}

/// `numpy.nextafter(x, 0.0)` — one ULP toward zero.
///
/// The radius shrink both estimators apply before counting, so the k-th
/// neighbour itself is excluded. Implemented on the bit pattern because `f64`
/// has no `next_down` in stable Rust: for a positive finite `x` the next
/// smaller `f64` is `from_bits(to_bits(x) − 1)`, which walks correctly across
/// the exponent boundary and lands on `0.0` from the smallest subnormal.
/// `0.0` is already at the target and stays put, matching numpy.
///
/// The negative branch is unreachable from this module (every argument is a
/// distance) but is written correctly rather than left as a copy of the positive
/// one: for a NEGATIVE `f64` the sign-magnitude layout means decrementing the
/// bits moves AWAY from zero, so toward-zero is `bits − 1` on the positive side
/// and `bits + 1` on the negative side.
fn next_toward_zero(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Column-major extraction of column `c` from a row-major `n × d` slice.
fn column(x: &[f64], n: usize, d: usize, c: usize) -> Vec<f64> {
    (0..n).map(|r| x[r * d + c]).collect()
}

/// `sklearn.preprocessing.scale(v, with_mean=False)` in place: divide by the
/// standard deviation ABOUT THE MEAN (`ddof = 0`), leaving the mean in place.
///
/// The "about the mean" part is the subtle bit — `with_mean=False` skips the
/// SUBTRACTION, not the mean's role in the variance, so this is `v / std(v)`
/// and not `v / rms(v)`. sklearn's `_handle_zeros_in_scale` then replaces a zero
/// scale with `1.0`, so a constant vector passes through unchanged rather than
/// becoming `NaN`.
///
/// The divisor comes from [`numpy_std`], numpy's PAIRWISE-summed standard
/// deviation, not from a sequential sum. That is not defensive precision: a
/// 1-ULP difference in this one divisor moves tied points across the
/// `nextafter(kth, 0)` radius boundary and shifted this estimator by 1.5% on the
/// oracle's tied column. [`super::numpy_rng`]'s section comment derives it.
fn scale_no_mean(v: &mut [f64], axis0: bool) {
    let mut scale = if axis0 {
        numpy_std_axis0(v)
    } else {
        numpy_std(v)
    };
    if scale == 0.0 {
        scale = 1.0;
    }
    for x in v.iter_mut() {
        *x /= scale;
    }
}

/// Count, for each point, how many OTHER points of a 1-D sample lie strictly
/// within that point's own radius — the `KDTree.query_radius(count_only=True)`
/// call both estimators make.
///
/// Returns the RAW counts, INCLUDING each point itself (numpy's `query_radius`
/// counts the query point, which is why `_compute_mi_cc` subtracts 1 and
/// `_compute_mi_cd` does not).
///
/// Exact, via one sort plus two binary searches per point: `query_radius` on a
/// 1-D KD-tree with the Chebyshev or Euclidean metric — which coincide in one
/// dimension — counts exactly the points satisfying `|x − v| <= r` (verified
/// against `KDTree.query_radius`: it counts the query point itself and its test
/// is inclusive).
///
/// ## The predicate is on the DISTANCE, never on `[v − r, v + r]`
/// Searching the interval endpoints instead is wrong here, and wrong by exactly
/// one count for almost every point — a bug this module hit before the oracle
/// caught it. The radius handed in is deliberately ONE ULP BELOW a real
/// neighbour distance (`next_toward_zero` of the k-th, so the k-th neighbour is
/// excluded), and `v + (kth − ulp)` rounds back up to `v + kth` — i.e. to the
/// k-th neighbour's own value — so the interval form includes the very point the
/// ULP shrink existed to exclude. Every count came out one too high, `mean
/// ψ(m_all)` too large by ~0.2, and a genuinely-positive mutual information
/// clamped to `0.0` by the `max(0, ·)`.
///
/// Both bounds below therefore evaluate the SAME subtraction the metric does.
/// Each predicate is monotone over the ascending order — `v − s` decreases and
/// `s − v` increases — so `partition_point` is still valid.
/// ## The comparison is on the PLAIN difference, and a squared one is WRONG here
/// Comparing `(s − v)² <= r²` instead — which is what sklearn's binary trees do
/// internally for the Euclidean metric, via their "reduced distance" — was tried
/// and REGRESSED `mutual_info_classif` by 2.7e-2, three orders worse than the
/// contract. The reason is that this one helper serves both estimators: the
/// continuous-continuous path queries CHEBYSHEV marginals, where sklearn's tree
/// has no squared form to reduce to, so a squared comparison agrees with sklearn
/// on neither path. The plain difference matches `mutual_info_classif` exactly on
/// every configuration in the oracle, which is the evidence that settles it.
fn radius_counts_1d(values: &[f64], radii: &[f64]) -> Vec<usize> {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    values
        .iter()
        .zip(radii)
        .map(|(&v, &r)| {
            let lo = sorted.partition_point(|&s| v - s > r);
            let hi = sorted.partition_point(|&s| s - v <= r);
            hi.saturating_sub(lo)
        })
        .collect()
}

/// The Chebyshev distance to each point's `k`-th nearest OTHER point in the 2-D
/// `(x, y)` plane — `NearestNeighbors(metric="chebyshev", n_neighbors=k)`
/// followed by `kneighbors()[0][:, -1]`.
///
/// `kneighbors()` with no argument queries the TRAINING set and EXCLUDES each
/// point itself, which is why the scan skips `i == j` rather than taking the
/// `k+1`-th distance.
///
/// Brute force `O(n²)`: exact, and the identified perf lever (module docs).
/// The k smallest distances are kept in a small insertion-sorted buffer rather
/// than by sorting all `n` — `k` is 3 by default, so this is `O(n·k)` extra work
/// against `O(n²)` distance evaluations.
fn knn_chebyshev_2d_kth(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::INFINITY; n];
    let mut best = vec![f64::INFINITY; k];
    for i in 0..n {
        best.iter_mut().for_each(|b| *b = f64::INFINITY);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = (x[i] - x[j]).abs();
            let dy = (y[i] - y[j]).abs();
            let dist = if dx > dy { dx } else { dy };
            if dist < best[k - 1] {
                // Insertion into the ascending k-buffer.
                let mut p = k - 1;
                while p > 0 && best[p - 1] > dist {
                    best[p] = best[p - 1];
                    p -= 1;
                }
                best[p] = dist;
            }
        }
        out[i] = best[k - 1];
    }
    out
}

/// `|a − b|` — the 1-D Euclidean/Chebyshev distance (they coincide in one
/// dimension), as sklearn's `KDTree` evaluates it.
///
/// The tree computes `sqrt(rdist)` where `rdist = (a − b)²`, and the two forms
/// are BIT-IDENTICAL: for IEEE-754 doubles `sqrt(fl(x²))` rounds back to `|x|`
/// exactly whenever `x²` neither overflows nor underflows. So the plain form is
/// kept because it is the one that says what it means. The BRUTE path does NOT
/// share this property — see [`knn_1d_kth_brute`], which is the whole reason
/// this module has two searches instead of one.
fn euclid_1d(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

/// The `k`-th nearest OTHER point in 1-D, as sklearn's KD-TREE path computes it:
/// the exact `|a − b|` metric, in `O(n log n + n·k)`.
///
/// In 1-D the sorted order makes this exact: the `k`-th nearest neighbour of a
/// sorted point is found by expanding left/right from its own position, which is
/// the classic two-pointer merge over the two monotone distance streams.
fn knn_1d_kth_tree(values: &[f64], k: usize) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    let mut out = vec![f64::INFINITY; n];
    for (pos, &orig) in order.iter().enumerate() {
        // Two pointers walking outward from `pos`, taking the nearer side each
        // step; after `k` steps the last distance taken is the k-th nearest.
        let mut lo = pos as isize - 1;
        let mut hi = pos + 1;
        let mut last = f64::INFINITY;
        for _ in 0..k {
            let dlo = if lo >= 0 {
                euclid_1d(sorted[pos], sorted[lo as usize])
            } else {
                f64::INFINITY
            };
            let dhi = if hi < n {
                euclid_1d(sorted[hi], sorted[pos])
            } else {
                f64::INFINITY
            };
            if dlo.is_infinite() && dhi.is_infinite() {
                break;
            }
            // `<=` so a left-side TIE is consumed first. With both sides equally
            // distant the k-th DISTANCE is the same either way, which is all
            // this function returns, so the tie-break is not observable — it is
            // fixed only so the walk is deterministic.
            if dlo <= dhi {
                last = dlo;
                lo -= 1;
            } else {
                last = dhi;
                hi += 1;
            }
        }
        out[orig] = last;
    }
    out
}

/// The `k`-th nearest OTHER point in 1-D, as sklearn's BRUTE path computes it:
/// the GEMM identity `a² − 2ab + b²`, clamped at zero, then `sqrt`.
///
/// ## Why this is not the same number as `|a − b|`
/// sklearn's brute force never forms the difference. `ArgKmin` (and the
/// `euclidean_distances` fallback behind it) computes the SQUARED distance
/// matrix as `‖x‖² − 2·xᵀy + ‖y‖²` — one BLAS product plus two norm additions —
/// and takes the square root only of the `k` values it selects. Each of those
/// three terms is rounded separately, so the result differs from `|a − b|` by a
/// few ULP: for `a − b ≈ 0.0539` this module measured `0.053905995835782816`
/// against `|a − b| = 0.0539059958357857`.
///
/// A few ULP is decisive here rather than negligible. The caller shrinks this
/// value by exactly ONE ULP to form the counting radius, so a distance that is
/// several ULP too large admits points sklearn's radius excludes, and `m_all`
/// changes by whole integers. On this crate's oracle it moved
/// `mutual_info_regression` by 1.2e-1 on the binned column — four orders past
/// the 1e-5 contract.
///
/// The arithmetic is BLAS-INDEPENDENT despite going through GEMM, which is what
/// makes it reproducible at all: the product has inner dimension `k = 1` here
/// (the sample is one-dimensional), so `xᵀy` is a SINGLE multiplication with
/// nothing to reassociate, and scaling it by `−2` is exact. The association of
/// the two remaining additions is sklearn's — `(‖x‖² + middle) + ‖y‖²` — and
/// `‖x‖²` is `np.einsum('ij,ij->i', X, X)`, i.e. `a·a` with one rounding.
///
/// `O(n²)`, which is not a concern because the caller only reaches this branch
/// for SMALL groups: [`knn_1d_kth`] dispatches here exactly when
/// `k >= count / 2`, and `k` is itself capped at `n_neighbors`, so `count` is at
/// most `2·n_neighbors + 1`.
fn knn_1d_kth_brute(values: &[f64], k: usize) -> Vec<f64> {
    let n = values.len();
    let sq: Vec<f64> = values.iter().map(|&v| v * v).collect();
    let mut dists: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
    let mut out = vec![f64::INFINITY; n];
    for i in 0..n {
        dists.clear();
        for j in 0..n {
            if i == j {
                continue;
            }
            // sklearn's term order: `X_norm_squared[i] + middle + Y_norm_squared[j]`.
            let middle = -2.0 * (values[i] * values[j]);
            let s = (sq[i] + middle) + sq[j];
            // `np.maximum(distances, 0)` — the identity can go slightly negative
            // for near-coincident points, and a negative `sqrt` would be NaN.
            let s = if s > 0.0 { s } else { 0.0 };
            dists.push(s.sqrt());
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        out[i] = dists[k - 1];
    }
    out
}

/// The Euclidean distance to each point's `k`-th nearest OTHER point in a 1-D
/// sample — `NearestNeighbors(n_neighbors=k).fit(c).kneighbors()[0][:, -1]`, as
/// `_compute_mi_cd` calls it within each label group.
///
/// ## The `algorithm='auto'` dispatch is OBSERVABLE, not an implementation detail
/// `NearestNeighbors` defaults to `algorithm='auto'` and picks its search at FIT
/// time. `NeighborsBase._fit` selects BRUTE when
///
/// ```text
/// metric == 'precomputed'  or  n_features > 15  or  n_neighbors >= n_samples // 2
/// ```
///
/// and a KD-tree otherwise. The sample here is always one-dimensional and the
/// metric is the default `minkowski(p=2)` (remapped to `euclidean`), so only the
/// NEIGHBOUR-COUNT clause can fire — and it fires constantly, because
/// `_compute_mi_cd` caps `k` at `count − 1` per label group: any group with
/// `count <= 2·n_neighbors + 1` takes the brute path.
///
/// The two paths return DIFFERENT last bits ([`knn_1d_kth_brute`] derives why),
/// and this estimator counts neighbours inside a radius one ULP below this
/// value, so following the dispatch is what makes the result match sklearn
/// instead of merely approximating it. Reproducing only the tree path left
/// `mutual_info_regression` wrong by 1.2e-1 on the oracle's binned column and by
/// 5.7e-4 on two ordinary ones — a discrepancy previously mis-attributed to
/// non-reproducible BLAS rounding, when it was this branch all along.
///
/// The `n_features > 15` clause is unreachable from here (the sample is 1-D) and
/// the `precomputed` clause has no mlrs analogue, so neither is reproduced.
fn knn_1d_kth(values: &[f64], k: usize) -> Vec<f64> {
    if k >= values.len() / 2 {
        knn_1d_kth_brute(values, k)
    } else {
        knn_1d_kth_tree(values, k)
    }
}

/// Kraskov mutual information between two CONTINUOUS 1-D samples —
/// sklearn's `_compute_mi_cc`.
///
/// ```text
/// MI = ψ(n) + ψ(k) − mean ψ(nx + 1) − mean ψ(ny + 1)
/// ```
///
/// where `nx`/`ny` count the points within the (ULP-shrunk) 2-D k-th-neighbour
/// radius in each marginal, minus the point itself.
fn compute_mi_cc(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();
    let radius: Vec<f64> = knn_chebyshev_2d_kth(x, y, k)
        .into_iter()
        .map(next_toward_zero)
        .collect();
    let nx = radius_counts_1d(x, &radius);
    let ny = radius_counts_1d(y, &radius);
    // `−1.0` removes the query point itself, then `+1` restores it inside ψ —
    // sklearn writes it as `nx = counts − 1.0` followed by `digamma(nx + 1)`,
    // i.e. `digamma(counts)`. Kept in sklearn's two-step form because the
    // cancellation is only exact for integer counts and reading the algebraic
    // shortcut back onto the paper formula is where a reader loses the thread.
    let mean_x = nx
        .iter()
        .map(|&c| digamma(c as f64 - 1.0 + 1.0))
        .sum::<f64>()
        / n as f64;
    let mean_y = ny
        .iter()
        .map(|&c| digamma(c as f64 - 1.0 + 1.0))
        .sum::<f64>()
        / n as f64;
    let mi = digamma(n as f64) + digamma(k as f64) - mean_x - mean_y;
    mi.max(0.0)
}

/// Ross mutual information between a CONTINUOUS sample `c` and a DISCRETE
/// label vector `d` — sklearn's `_compute_mi_cd`.
///
/// ```text
/// MI = ψ(n') + mean ψ(k_all) − mean ψ(label_counts) − mean ψ(m_all)
/// ```
///
/// Three details carry the whole implementation:
///
/// * `k` is per-GROUP: `min(n_neighbors, count − 1)`, so a small class uses a
///   smaller neighbourhood, and a SINGLETON class contributes no radius at all;
/// * points whose label is unique are then DROPPED entirely (`label_counts > 1`)
///   and `n'` is the surviving count, not `n`. Dropping them after computing the
///   radii — sklearn's order — is what makes `m_all`'s radius search run over
///   the FILTERED sample;
/// * the per-group radius goes through [`knn_1d_kth`], which follows sklearn's
///   `algorithm='auto'` brute-vs-tree dispatch. That is a consequence of the
///   first detail: capping `k` at `count − 1` is exactly what pushes small
///   groups over `k >= count / 2` and onto the brute path, whose distances
///   differ from the tree's in the last bits.
fn compute_mi_cd(c: &[f64], labels: &[u32], k_neighbors: usize) -> f64 {
    let n = c.len();
    let mut radius = vec![0.0f64; n];
    let mut label_counts = vec![0usize; n];
    let mut k_all = vec![0.0f64; n];

    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        groups.entry(l).or_default().push(i);
    }
    // Iterating a HashMap is order-dependent, but every write below is to a
    // per-INDEX slot with no cross-group accumulation, so the result is
    // order-independent. (`np.unique(d)`'s sorted order is therefore not needed
    // here, unlike in `class_indices`.)
    for members in groups.values() {
        let count = members.len();
        for &i in members {
            label_counts[i] = count;
        }
        if count > 1 {
            let k = k_neighbors.min(count - 1);
            let vals: Vec<f64> = members.iter().map(|&i| c[i]).collect();
            let kth = knn_1d_kth(&vals, k);
            for (slot, &i) in members.iter().enumerate() {
                radius[i] = next_toward_zero(kth[slot]);
                k_all[i] = k as f64;
            }
        }
    }

    // Drop the unique-label points.
    let keep: Vec<usize> = (0..n).filter(|&i| label_counts[i] > 1).collect();
    let n_kept = keep.len();
    if n_kept == 0 {
        // Every label occurs once: sklearn's `digamma(0)` path. numpy returns
        // `inf` for `digamma(0)` and the means over empty slices are `NaN`, so
        // the whole expression is `NaN` and `max(0, NaN)` is... `0` in Python
        // (`max` compares `NaN > 0` as False and returns the first argument).
        // Returning `0.0` reproduces that without relying on a NaN comparison.
        return 0.0;
    }
    let c_kept: Vec<f64> = keep.iter().map(|&i| c[i]).collect();
    let r_kept: Vec<f64> = keep.iter().map(|&i| radius[i]).collect();
    let m_all = radius_counts_1d(&c_kept, &r_kept);

    let inv = 1.0 / n_kept as f64;
    let mean_k = keep.iter().map(|&i| digamma(k_all[i])).sum::<f64>() * inv;
    let mean_lc = keep
        .iter()
        .map(|&i| digamma(label_counts[i] as f64))
        .sum::<f64>()
        * inv;
    let mean_m = m_all.iter().map(|&m| digamma(m as f64)).sum::<f64>() * inv;
    let mi = digamma(n_kept as f64) + mean_k - mean_lc - mean_m;
    mi.max(0.0)
}

/// `sklearn.metrics.mutual_info_score(x, y)` for two DISCRETE samples — the
/// plug-in contingency-table estimator, used when a feature and the target are
/// both discrete.
///
/// ```text
/// MI = Σ_ij (n_ij/n) · ln( (n_ij·n) / (a_i·b_j) )
/// ```
///
/// over the non-zero cells only. sklearn evaluates the log as
/// `log(nij) − log(a_i) − log(b_j) + log(n)` via its `_nonzero` contingency
/// path; the grouped form here is algebraically identical and, with all four
/// terms `O(ln n)`, agrees to the last bits at any `n` a feature matrix reaches.
fn mutual_info_discrete(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mut cells: HashMap<(u64, u64), usize> = HashMap::new();
    let mut row: HashMap<u64, usize> = HashMap::new();
    let mut col: HashMap<u64, usize> = HashMap::new();
    for i in 0..n {
        // Keyed on the bit pattern so the grouping is exact float equality,
        // which is what `np.unique` gives sklearn's contingency table.
        let (a, b) = (x[i].to_bits(), y[i].to_bits());
        *cells.entry((a, b)).or_insert(0) += 1;
        *row.entry(a).or_insert(0) += 1;
        *col.entry(b).or_insert(0) += 1;
    }
    let n_f = n as f64;
    let mut mi = 0.0;
    for (&(a, b), &nij) in cells.iter() {
        let ai = row[&a] as f64;
        let bj = col[&b] as f64;
        let nij_f = nij as f64;
        mi += (nij_f / n_f) * ((nij_f * n_f) / (ai * bj)).ln();
    }
    // sklearn clips the result at 0 (`np.clip(mi.sum(), 0.0, None)`); a
    // single-valued x or y gives an exact 0 that can round slightly negative.
    mi.max(0.0)
}

/// sklearn's `_estimate_mi` — the shared driver behind both public functions.
///
/// Order of operations is sklearn's exactly, and it is observable because the
/// RNG stream is consumed along the way:
///
/// 1. resolve the discrete mask;
/// 2. if ANY column is continuous, `scale(X[:, cont], with_mean=False)` then add
///    `1e-10 · max(1, mean|X_cont|) · N(0,1)` — with the mean taken over the
///    WHOLE continuous block per column, and ONE `standard_normal` draw of shape
///    `(n, n_cont)` filled C-order;
/// 3. if the target is continuous, `scale(y, with_mean=False)` then add
///    `1e-10 · max(1, mean|y|) · N(0,1)` from the SAME stream, continuing where
///    step 2 left off;
/// 4. per column, dispatch on `(feature_discrete, target_discrete)`.
///
/// Step 3 drawing from the same generator AFTER step 2 is why the two noise
/// blocks cannot be generated independently, and step 2 drawing a single
/// `(n, n_cont)` block — not one column at a time — is why a per-column loop
/// over the RNG would desynchronise from sklearn on the second column onward.
fn estimate_mi(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    params: &MutualInfoParams,
    discrete_target: bool,
) -> Result<Vec<f64>, AlgoError> {
    if params.n_neighbors == 0 {
        return Err(AlgoError::InvalidSelectorParam {
            estimator: "mutual_info",
            param: "n_neighbors",
            value: 0.0,
            reason: "must be >= 1",
        });
    }
    if params.n_jobs == Some(0) {
        return Err(AlgoError::InvalidSelectorParam {
            estimator: "mutual_info",
            param: "n_jobs",
            value: 0.0,
            reason: "must be non-zero (sklearn: None, -1, or a positive count)",
        });
    }
    if x.len() != n * d || y.len() != n {
        return Err(AlgoError::Prim(mlrs_core::PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        }));
    }
    let discrete_mask = params.discrete_features.resolve(d)?;
    let cont: Vec<usize> = (0..d).filter(|&c| !discrete_mask[c]).collect();

    let mut rng = NumpyRandomState::new(params.random_state.unwrap_or(0));
    // Columns are held column-major from here: every downstream step is a
    // single-column operation, and the scale/noise passes need a contiguous
    // column anyway.
    let mut cols: Vec<Vec<f64>> = (0..d).map(|c| column(x, n, d, c)).collect();

    if !cont.is_empty() {
        for &c in cont.iter() {
            // A COLUMN of the 2-D design: the axis-0 (sequential) reduction.
            scale_no_mean(&mut cols[c], true);
        }
        // `means = np.maximum(1, np.mean(np.abs(X[:, cont]), axis=0))` — one
        // value per CONTINUOUS column, computed AFTER the scaling.
        let means: Vec<f64> = cont
            .iter()
            .map(|&c| {
                let abs: Vec<f64> = cols[c].iter().map(|v| v.abs()).collect();
                numpy_mean_axis0(&abs).max(1.0)
            })
            .collect();
        // ONE `(n, n_cont)` C-order draw, so the noise for row r column j is
        // draw[r * n_cont + j] — matching numpy's row-major fill.
        let noise = rng.standard_normal_vec(n * cont.len());
        for (j, &c) in cont.iter().enumerate() {
            for r in 0..n {
                cols[c][r] += 1e-10 * means[j] * noise[r * cont.len() + j];
            }
        }
    }

    let mut y_work = y.to_vec();
    if !discrete_target {
        // The 1-D target: the contiguous (pairwise) reduction.
        scale_no_mean(&mut y_work, false);
        let abs: Vec<f64> = y_work.iter().map(|v| v.abs()).collect();
        let m = numpy_mean(&abs).max(1.0);
        let noise = rng.standard_normal_vec(n);
        for r in 0..n {
            y_work[r] += 1e-10 * m * noise[r];
        }
    }

    // The discrete target's class indices, needed by every continuous column.
    let (target_labels, _) = if discrete_target {
        super::score::class_indices(&y_work)
    } else {
        (Vec::new(), Vec::new())
    };

    let per_column = |c: usize| -> f64 {
        match (discrete_mask[c], discrete_target) {
            (true, true) => mutual_info_discrete(&cols[c], &y_work),
            (true, false) => {
                // Feature discrete, target continuous: sklearn swaps the
                // arguments (`_compute_mi_cd(y, x, k)`) so the CONTINUOUS one is
                // the sample and the DISCRETE one the label.
                let (labels, _) = super::score::class_indices(&cols[c]);
                compute_mi_cd(&y_work, &labels, params.n_neighbors)
            }
            (false, true) => compute_mi_cd(&cols[c], &target_labels, params.n_neighbors),
            (false, false) => compute_mi_cc(&cols[c], &y_work, params.n_neighbors),
        }
    };

    // The columns are independent, so `n_jobs` splits them with no shared state
    // — the `mbsgd` OvR fan-out shape, not the barrier-synchronised pool shape.
    let units = match params.n_jobs {
        None | Some(1) => 1,
        // sklearn's `-1` means "all processors"; `usize` cannot hold `-1`, so
        // the shim maps it to the machine's unit count before it reaches here.
        Some(k) => k.min(d).max(1),
    };
    if units <= 1 {
        return Ok((0..d).map(per_column).collect());
    }
    let chunk = d.div_ceil(units);
    let parts: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..units)
            .filter_map(|u| {
                let c0 = u * chunk;
                if c0 >= d {
                    return None;
                }
                let c1 = (c0 + chunk).min(d);
                let f = &per_column;
                Some(scope.spawn(move || (c0..c1).map(f).collect::<Vec<f64>>()))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("mutual_info column worker panicked"))
            .collect()
    });
    Ok(parts.into_iter().flatten().collect())
}

/// `sklearn.feature_selection.mutual_info_regression(X, y, ...)` — mutual
/// information between each column and a CONTINUOUS target. Scores only (no
/// p-values).
pub fn mutual_info_regression(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    params: &MutualInfoParams,
) -> Result<Vec<f64>, AlgoError> {
    estimate_mi(x, y, n, d, params, false)
}

/// `sklearn.feature_selection.mutual_info_classif(X, y, ...)` — mutual
/// information between each column and a DISCRETE target. Scores only.
pub fn mutual_info_classif(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    params: &MutualInfoParams,
) -> Result<Vec<f64>, AlgoError> {
    estimate_mi(x, y, n, d, params, true)
}
