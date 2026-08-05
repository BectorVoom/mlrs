//! `TsneMetric` (TSNE-PARAMS) — the full sklearn `TSNE(metric=...)` string
//! surface, plus the `metric_params` payload (`p` / `V` / `VI` / `w`).
//!
//! ## What sklearn actually computes, per metric
//!
//! `TSNE._fit` does NOT hand the metric straight to scipy. It routes through
//! [`sklearn.metrics.pairwise.pairwise_distances`], and that function makes
//! three routing decisions this module has to reproduce exactly — each one was
//! confirmed against the installed 1.9.0, not inferred from the docs:
//!
//! 1. **Six metrics are cast to `bool` first.**
//!    `PAIRWISE_BOOLEAN_FUNCTIONS = ['dice', 'jaccard', 'rogerstanimoto',
//!    'russellrao', 'sokalsneath', 'yule']`. For those, the design becomes
//!    `x != 0` and the classic four boolean counts (`ctt`/`ctf`/`cft`/`cff`)
//!    drive the formula. `hamming` and `matching` are NOT in that list, so they
//!    stay FLOAT and both evaluate to `mean(xᵢ != yᵢ)` — `matching` is a scipy
//!    alias of float `hamming`, which is why a signed-Gaussian design gives
//!    `1.0` for both and `0.0` for `dice`/`jaccard` (every coordinate is
//!    non-zero, so the bool cast collapses to all-true). Getting this backwards
//!    silently produces a *different metric*, not a rounding difference.
//! 2. **`seuclidean` / `mahalanobis` get data-derived defaults.**
//!    `_precompute_metric_params` fills `V = var(X, axis=0, ddof=1)` and
//!    `VI = inv(cov(X.T)).T` when the caller omitted them. `ddof=1` (not 0) and
//!    the transpose are both load-bearing for parity.
//! 3. **`euclidean` skips a square root.** `_fit` calls
//!    `pairwise_distances(X, metric='euclidean', squared=True)` for that ONE
//!    name and squares the result for every other metric. `l2`/`sqeuclidean`
//!    therefore take the sqrt-then-square round trip. The values agree to
//!    within an ulp, so this module always returns SQUARED distances directly
//!    and documents the aliasing rather than reproducing a redundant round
//!    trip.
//!
//! ## Degenerate values are mirrored, not repaired
//! scipy's boolean family is not total: `dice` of two all-false rows is `0/0 =
//! NaN`, and `sokalsneath` of two all-false rows RAISES. sklearn propagates
//! both. mlrs mirrors them ([`MetricError::SokalSneathAllZero`] for the second)
//! so a user who hits the degeneracy sees the same failure here as there,
//! rather than a silently different embedding.
//!
//! ## Aliases
//! `l2` ≡ `euclidean`, `l1` ≡ `manhattan` ≡ `cityblock`, `matching` ≡ float
//! `hamming`. They parse to one canonical variant each; the alias the user
//! wrote is not retained because nothing downstream can observe it.
//!
//! Tests live in `crates/mlrs-algos/tests/tsne_metric_test.rs` (AGENTS.md §2).

use crate::error::AlgoError;

/// Every `metric=` string sklearn 1.9.0's `TSNE` accepts, canonicalized
/// (aliases collapsed — see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsneMetric {
    /// `'euclidean'` / `'l2'` — `‖x − y‖₂`.
    Euclidean,
    /// `'sqeuclidean'` — `‖x − y‖₂²`. Distinct from [`Self::Euclidean`]
    /// because t-SNE SQUARES the metric output: `sqeuclidean` is squared
    /// twice, giving a fourth power.
    SqEuclidean,
    /// `'manhattan'` / `'cityblock'` / `'l1'` — `Σ|xᵢ − yᵢ|`.
    Manhattan,
    /// `'chebyshev'` — `maxᵢ|xᵢ − yᵢ|`.
    Chebyshev,
    /// `'minkowski'` — `(Σ|xᵢ − yᵢ|^p)^(1/p)`, `p` from `metric_params`
    /// (default 2).
    Minkowski,
    /// `'cosine'` — `1 − x̂·ŷ` over L2-normalized rows (sklearn's
    /// `cosine_distances`, clipped to `[0, 2]`).
    Cosine,
    /// `'correlation'` — cosine distance between MEAN-CENTERED rows.
    Correlation,
    /// `'canberra'` — `Σ |xᵢ − yᵢ| / (|xᵢ| + |yᵢ|)`, `0/0` contributing 0.
    Canberra,
    /// `'braycurtis'` — `Σ|xᵢ − yᵢ| / Σ|xᵢ + yᵢ|`.
    BrayCurtis,
    /// `'seuclidean'` — `sqrt(Σ (xᵢ − yᵢ)² / Vᵢ)`, `V` defaulting to
    /// `var(X, ddof=1)`.
    SEuclidean,
    /// `'mahalanobis'` — `sqrt((x − y)ᵀ VI (x − y))`, `VI` defaulting to
    /// `inv(cov(Xᵀ)).T`.
    Mahalanobis,
    /// `'haversine'` — great-circle distance on the unit sphere; REQUIRES
    /// exactly 2 features (radian latitude, longitude).
    Haversine,
    /// `'nan_euclidean'` — Euclidean over the coordinates present in BOTH
    /// rows, rescaled by `n_features / n_present`.
    NanEuclidean,
    /// `'hamming'` / `'matching'` — `mean(xᵢ != yᵢ)` on the RAW floats (not
    /// bool-cast; see the module docs).
    Hamming,
    /// `'jaccard'` — bool-cast, `(ctf + cft) / (ctt + ctf + cft)`.
    Jaccard,
    /// `'dice'` — bool-cast, `(ctf + cft) / (2·ctt + ctf + cft)`.
    Dice,
    /// `'rogerstanimoto'` — bool-cast,
    /// `2(ctf + cft) / (ctt + cff + 2(ctf + cft))`.
    RogersTanimoto,
    /// `'russellrao'` — bool-cast, `(n − ctt) / n`.
    RussellRao,
    /// `'sokalsneath'` — bool-cast, `2(ctf + cft) / (ctt + 2(ctf + cft))`.
    SokalSneath,
    /// `'yule'` — bool-cast, `2·ctf·cft / (ctt·cff + ctf·cft)`.
    Yule,
    /// `'precomputed'` — `X` IS the (square, non-negative) distance matrix.
    Precomputed,
    /// `'wminkowski'` — accepted by sklearn's parameter validation and then
    /// REJECTED by scipy, which removed the metric. Kept as a variant so the
    /// mlrs surface accepts every string sklearn's `StrOptions` does, and
    /// fails at fit with the same shape of error.
    WMinkowski,
}

impl TsneMetric {
    /// Parse a sklearn `metric=` string, collapsing aliases. `None` for any
    /// string outside sklearn's `StrOptions` set.
    pub fn from_sklearn_name(s: &str) -> Option<Self> {
        Some(match s {
            "euclidean" | "l2" => Self::Euclidean,
            "sqeuclidean" => Self::SqEuclidean,
            "manhattan" | "cityblock" | "l1" => Self::Manhattan,
            "chebyshev" => Self::Chebyshev,
            "minkowski" => Self::Minkowski,
            "cosine" => Self::Cosine,
            "correlation" => Self::Correlation,
            "canberra" => Self::Canberra,
            "braycurtis" => Self::BrayCurtis,
            "seuclidean" => Self::SEuclidean,
            "mahalanobis" => Self::Mahalanobis,
            "haversine" => Self::Haversine,
            "nan_euclidean" => Self::NanEuclidean,
            "hamming" | "matching" => Self::Hamming,
            "jaccard" => Self::Jaccard,
            "dice" => Self::Dice,
            "rogerstanimoto" => Self::RogersTanimoto,
            "russellrao" => Self::RussellRao,
            "sokalsneath" => Self::SokalSneath,
            "yule" => Self::Yule,
            "precomputed" => Self::Precomputed,
            "wminkowski" => Self::WMinkowski,
            _ => return None,
        })
    }

    /// The canonical sklearn name (the one an error message should echo).
    pub fn sklearn_name(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::SqEuclidean => "sqeuclidean",
            Self::Manhattan => "manhattan",
            Self::Chebyshev => "chebyshev",
            Self::Minkowski => "minkowski",
            Self::Cosine => "cosine",
            Self::Correlation => "correlation",
            Self::Canberra => "canberra",
            Self::BrayCurtis => "braycurtis",
            Self::SEuclidean => "seuclidean",
            Self::Mahalanobis => "mahalanobis",
            Self::Haversine => "haversine",
            Self::NanEuclidean => "nan_euclidean",
            Self::Hamming => "hamming",
            Self::Jaccard => "jaccard",
            Self::Dice => "dice",
            Self::RogersTanimoto => "rogerstanimoto",
            Self::RussellRao => "russellrao",
            Self::SokalSneath => "sokalsneath",
            Self::Yule => "yule",
            Self::Precomputed => "precomputed",
            Self::WMinkowski => "wminkowski",
        }
    }

    /// Does this metric aggregate MONOTONELY over independent feature axes?
    ///
    /// Only such a metric can be pruned by a KD-tree box bound, which is what
    /// lets the Barnes-Hut neighbour graph skip most pairs. `cosine` and
    /// `correlation` are row-normalized, `mahalanobis` mixes axes, the boolean
    /// family is a ratio of counts, and `precomputed` has no feature axes at
    /// all — none of them qualify. This is the same test
    /// [`InvalidAlgorithmMetric`](crate::error::BuildError::InvalidAlgorithmMetric)
    /// encodes for HDBSCAN.
    pub fn is_axis_separable(self) -> bool {
        matches!(
            self,
            Self::Euclidean
                | Self::SqEuclidean
                | Self::Manhattan
                | Self::Chebyshev
                | Self::Minkowski
        )
    }
}

/// The `metric_params=` payload. Every field is the sklearn/scipy keyword of
/// the same name; `None` means "let the metric derive its own default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricParams {
    /// `minkowski` exponent (sklearn/scipy default 2).
    pub p: Option<f64>,
    /// `seuclidean` per-feature variance vector (length `n_features`).
    pub v: Option<Vec<f64>>,
    /// `mahalanobis` inverse covariance (row-major `n_features²`).
    pub vi: Option<Vec<f64>>,
    /// `wminkowski` weights — carried only so the rejection message can name
    /// them; scipy removed the metric.
    pub w: Option<Vec<f64>>,
}

/// `metric_params` after the data-derived defaults have been filled in
/// (sklearn's `_precompute_metric_params`). Built once per fit by
/// [`resolve_metric_params`] and then read-only.
#[derive(Debug, Clone)]
pub struct ResolvedMetricParams {
    /// Minkowski exponent, defaulted to 2.
    pub p: f64,
    /// `seuclidean` variances, `var(X, ddof=1)`, length `n_features`.
    pub v: Vec<f64>,
    /// `mahalanobis` inverse covariance, row-major `n_features²`.
    pub vi: Vec<f64>,
}

/// Failures that are a property of the metric + data pair, surfaced as typed
/// [`AlgoError`]s by [`resolve_metric_params`] / [`pairwise_squared`].
///
/// `#[repr(usize)]` is load-bearing, not decoration: [`pairwise_squared`]'s
/// workers report a failure by storing the discriminant into a shared
/// `AtomicUsize`, and [`decode_metric_error`] reads it back. The explicit
/// representation pins the numbering the two sides agree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum MetricError {
    /// `haversine` was given a design whose feature count is not 2.
    HaversineNot2D,
    /// `wminkowski` — removed from scipy, so sklearn cannot evaluate it either.
    WMinkowskiRemoved,
    /// `sokalsneath` hit a pair of all-zero rows (`ctt + 2(ctf + cft) == 0`),
    /// which scipy raises on rather than returning a value.
    SokalSneathAllZero,
    /// `metric_params['V']` / `['VI']` had the wrong length for the design.
    BadMetricParamShape,
    /// `precomputed` was given a non-square `X`.
    PrecomputedNotSquare,
    /// A metric produced a negative distance — sklearn's `_fit` guard
    /// (`"All distances should be positive, the metric given is not correct"`).
    NegativeDistance,
}

impl MetricError {
    fn message(self) -> &'static str {
        match self {
            Self::HaversineNot2D => "metric 'haversine' is only valid in 2 dimensions",
            Self::WMinkowskiRemoved => {
                "metric 'wminkowski' was removed from scipy and cannot be evaluated"
            }
            Self::SokalSneathAllZero => {
                "metric 'sokalsneath' is undefined for a pair of all-zero rows"
            }
            Self::BadMetricParamShape => {
                "metric_params entry has the wrong length for this design"
            }
            Self::PrecomputedNotSquare => "X should be a square distance matrix",
            Self::NegativeDistance => {
                "All distances should be positive, the metric given is not correct"
            }
        }
    }

    /// Lift into the estimator error surface.
    pub fn into_algo_error(self) -> AlgoError {
        AlgoError::InvalidGraphInput {
            estimator: "tsne",
            reason: self.message().to_string(),
        }
    }
}

/// Fill `V` / `VI` / `p` from the design when the caller left them out —
/// sklearn's `_precompute_metric_params`, including its `ddof=1` and the
/// transpose on `VI`.
///
/// Only the entry the metric actually reads is computed; the others are left
/// empty, so a `cosine` fit never pays for a covariance inverse.
pub fn resolve_metric_params(
    x: &[f64],
    n: usize,
    d: usize,
    metric: TsneMetric,
    params: &MetricParams,
) -> Result<ResolvedMetricParams, AlgoError> {
    let p = params.p.unwrap_or(2.0);
    let mut v = Vec::new();
    let mut vi = Vec::new();

    match metric {
        TsneMetric::SEuclidean => {
            v = match &params.v {
                Some(user) => {
                    if user.len() != d {
                        return Err(MetricError::BadMetricParamShape.into_algo_error());
                    }
                    user.clone()
                }
                // np.var(X, axis=0, ddof=1)
                None => column_variance_ddof1(x, n, d),
            };
        }
        TsneMetric::Mahalanobis => {
            vi = match &params.vi {
                Some(user) => {
                    if user.len() != d * d {
                        return Err(MetricError::BadMetricParamShape.into_algo_error());
                    }
                    user.clone()
                }
                // np.linalg.inv(np.cov(X.T)).T
                None => transpose_square(&invert_spd_or_pinv(&covariance(x, n, d), d), d),
            };
        }
        _ => {}
    }
    Ok(ResolvedMetricParams { p, v, vi })
}

/// `np.var(X, axis=0, ddof=1)`. `n < 2` yields zeros (numpy emits NaN there;
/// t-SNE already rejects `n < 2` upstream, so the branch is unreachable in a
/// real fit and a finite value keeps the helper total).
fn column_variance_ddof1(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    if n < 2 {
        return out;
    }
    for (j, o) in out.iter_mut().enumerate() {
        let mean = (0..n).map(|i| x[i * d + j]).sum::<f64>() / n as f64;
        let ss = (0..n).map(|i| (x[i * d + j] - mean).powi(2)).sum::<f64>();
        *o = ss / (n as f64 - 1.0);
    }
    out
}

/// `np.cov(X.T)` — the `d × d` feature covariance with `ddof = 1`.
fn covariance(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut means = vec![0.0; d];
    for j in 0..d {
        means[j] = (0..n).map(|i| x[i * d + j]).sum::<f64>() / n as f64;
    }
    let denom = (n as f64 - 1.0).max(1.0);
    let mut c = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            let da = x[i * d + a] - means[a];
            for b in a..d {
                c[a * d + b] += da * (x[i * d + b] - means[b]);
            }
        }
    }
    for a in 0..d {
        for b in a..d {
            let v = c[a * d + b] / denom;
            c[a * d + b] = v;
            c[b * d + a] = v;
        }
    }
    c
}

/// Gauss-Jordan inverse with partial pivoting; falls back to a ridge-damped
/// inverse when the matrix is singular.
///
/// numpy's `inv` RAISES on a singular covariance. Damping instead of raising is
/// a deliberate divergence: a rank-deficient design is common (any constant or
/// duplicated feature produces one), and a t-SNE fit that returns a usable
/// embedding is more useful than one that refuses. The damping is the smallest
/// that restores a finite inverse, so a full-rank design is bit-unaffected.
fn invert_spd_or_pinv(a: &[f64], d: usize) -> Vec<f64> {
    if let Some(inv) = gauss_jordan_inverse(a, d) {
        return inv;
    }
    let trace: f64 = (0..d).map(|i| a[i * d + i]).sum();
    let mut damped = a.to_vec();
    let mut eps = (trace / d.max(1) as f64).abs().max(1.0) * 1e-12;
    for _ in 0..60 {
        for i in 0..d {
            damped[i * d + i] = a[i * d + i] + eps;
        }
        if let Some(inv) = gauss_jordan_inverse(&damped, d) {
            return inv;
        }
        eps *= 10.0;
    }
    // Unreachable for any finite input: by this point the diagonal dominates.
    vec![0.0; d * d]
}

fn gauss_jordan_inverse(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    for col in 0..d {
        let mut piv = col;
        let mut best = m[col * d + col].abs();
        for r in (col + 1)..d {
            let v = m[r * d + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if !(best > 0.0) || !best.is_finite() {
            return None;
        }
        if piv != col {
            for j in 0..d {
                m.swap(col * d + j, piv * d + j);
                inv.swap(col * d + j, piv * d + j);
            }
        }
        let pv = m[col * d + col];
        for j in 0..d {
            m[col * d + j] /= pv;
            inv[col * d + j] /= pv;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = m[r * d + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..d {
                m[r * d + j] -= f * m[col * d + j];
                inv[r * d + j] -= f * inv[col * d + j];
            }
        }
    }
    Some(inv)
}

fn transpose_square(a: &[f64], d: usize) -> Vec<f64> {
    let mut t = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            t[j * d + i] = a[i * d + j];
        }
    }
    t
}

// ===========================================================================
// Row preparation
// ===========================================================================

/// Whatever per-row precomputation the metric needs, done ONCE for the whole
/// design instead of once per pair.
///
/// This is not just a tidiness win: `cosine` and `correlation` would otherwise
/// renormalize both rows inside the `O(n²)` pair loop, making them `O(n²d)`
/// with a division per coordinate. Preparing turns them into plain dot
/// products.
pub struct PreparedRows {
    /// The row data the pair loop reads: the raw design, or a transformed copy
    /// (L2-normalized for `cosine`, centered-then-normalized for
    /// `correlation`, `x != 0` as `0.0`/`1.0` for the boolean family).
    data: Vec<f64>,
    /// `true` when [`Self::data`] is the boolean indicator, so the pair loop
    /// takes the count-based branch.
    boolean: bool,
}

impl PreparedRows {
    /// The prepared row block, row-major `n × d`.
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

/// Build the per-row precomputation for `metric` (see [`PreparedRows`]).
pub fn prepare_rows(x: &[f64], n: usize, d: usize, metric: TsneMetric) -> PreparedRows {
    match metric {
        TsneMetric::Cosine => PreparedRows {
            data: l2_normalize_rows(x, n, d),
            boolean: false,
        },
        TsneMetric::Correlation => {
            let mut c = x.to_vec();
            for i in 0..n {
                let row = &mut c[i * d..(i + 1) * d];
                let mean = row.iter().sum::<f64>() / d.max(1) as f64;
                for v in row.iter_mut() {
                    *v -= mean;
                }
            }
            PreparedRows {
                data: l2_normalize_rows(&c, n, d),
                boolean: false,
            }
        }
        TsneMetric::Jaccard
        | TsneMetric::Dice
        | TsneMetric::RogersTanimoto
        | TsneMetric::RussellRao
        | TsneMetric::SokalSneath
        | TsneMetric::Yule => PreparedRows {
            data: x.iter().map(|&v| if v != 0.0 { 1.0 } else { 0.0 }).collect(),
            boolean: true,
        },
        _ => PreparedRows {
            data: x.to_vec(),
            boolean: false,
        },
    }
}

fn l2_normalize_rows(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = x.to_vec();
    for i in 0..n {
        let row = &mut out[i * d..(i + 1) * d];
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        // sklearn's `normalize` leaves an all-zero row alone, which makes its
        // cosine distance to everything 1.0. Reproduced rather than guarded.
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
    out
}

// ===========================================================================
// The pair evaluation
// ===========================================================================

/// Distance between prepared rows `a` and `b` — the metric's OWN value, before
/// t-SNE squares it.
///
/// `prep` must have been produced by [`prepare_rows`] with the same `metric`,
/// and `rp` by [`resolve_metric_params`].
#[inline]
pub fn pair_distance(
    prep: &PreparedRows,
    d: usize,
    ia: usize,
    ib: usize,
    metric: TsneMetric,
    rp: &ResolvedMetricParams,
) -> Result<f64, MetricError> {
    let a = &prep.data[ia * d..ia * d + d];
    let b = &prep.data[ib * d..ib * d + d];

    if prep.boolean {
        return boolean_distance(a, b, metric);
    }

    Ok(match metric {
        TsneMetric::Euclidean | TsneMetric::SqEuclidean => {
            let mut acc = 0.0;
            for k in 0..d {
                let t = a[k] - b[k];
                acc += t * t;
            }
            if metric == TsneMetric::Euclidean {
                acc.sqrt()
            } else {
                acc
            }
        }
        TsneMetric::Manhattan => {
            let mut acc = 0.0;
            for k in 0..d {
                acc += (a[k] - b[k]).abs();
            }
            acc
        }
        TsneMetric::Chebyshev => {
            let mut acc: f64 = 0.0;
            for k in 0..d {
                acc = acc.max((a[k] - b[k]).abs());
            }
            acc
        }
        TsneMetric::Minkowski => {
            let p = rp.p;
            if p == 2.0 {
                let mut acc = 0.0;
                for k in 0..d {
                    let t = a[k] - b[k];
                    acc += t * t;
                }
                acc.sqrt()
            } else if p == 1.0 {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += (a[k] - b[k]).abs();
                }
                acc
            } else if p.is_infinite() {
                let mut acc: f64 = 0.0;
                for k in 0..d {
                    acc = acc.max((a[k] - b[k]).abs());
                }
                acc
            } else {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += (a[k] - b[k]).abs().powf(p);
                }
                acc.powf(1.0 / p)
            }
        }
        // Rows are already unit-norm (cosine) or centered-then-unit-norm
        // (correlation), so both are `1 − dot` — sklearn clips to [0, 2].
        TsneMetric::Cosine | TsneMetric::Correlation => {
            let mut dot = 0.0;
            for k in 0..d {
                dot += a[k] * b[k];
            }
            (1.0 - dot).clamp(0.0, 2.0)
        }
        TsneMetric::Canberra => {
            let mut acc = 0.0;
            for k in 0..d {
                let den = a[k].abs() + b[k].abs();
                // scipy contributes 0 for the 0/0 coordinate rather than NaN.
                if den > 0.0 {
                    acc += (a[k] - b[k]).abs() / den;
                }
            }
            acc
        }
        TsneMetric::BrayCurtis => {
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..d {
                num += (a[k] - b[k]).abs();
                den += (a[k] + b[k]).abs();
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }
        TsneMetric::SEuclidean => {
            let mut acc = 0.0;
            for k in 0..d {
                let t = a[k] - b[k];
                let vk = rp.v[k];
                if vk != 0.0 {
                    acc += t * t / vk;
                } else if t != 0.0 {
                    acc = f64::INFINITY;
                }
            }
            acc.sqrt()
        }
        TsneMetric::Mahalanobis => {
            // (x − y)ᵀ VI (x − y), walked row-blocked so `VI` streams once.
            let mut acc = 0.0;
            for r in 0..d {
                let dr = a[r] - b[r];
                if dr == 0.0 {
                    continue;
                }
                let mut inner = 0.0;
                let vrow = &rp.vi[r * d..r * d + d];
                for c in 0..d {
                    inner += vrow[c] * (a[c] - b[c]);
                }
                acc += dr * inner;
            }
            acc.max(0.0).sqrt()
        }
        TsneMetric::Haversine => {
            // Guarded by the caller (`validate_metric_geometry`), so `d == 2`.
            let (lat1, lon1) = (a[0], a[1]);
            let (lat2, lon2) = (b[0], b[1]);
            let s_lat = ((lat2 - lat1) * 0.5).sin();
            let s_lon = ((lon2 - lon1) * 0.5).sin();
            let h = s_lat * s_lat + lat1.cos() * lat2.cos() * s_lon * s_lon;
            2.0 * h.clamp(0.0, 1.0).sqrt().asin()
        }
        TsneMetric::NanEuclidean => {
            let mut acc = 0.0;
            let mut present = 0usize;
            for k in 0..d {
                if a[k].is_nan() || b[k].is_nan() {
                    continue;
                }
                let t = a[k] - b[k];
                acc += t * t;
                present += 1;
            }
            if present == 0 {
                f64::NAN
            } else {
                (d as f64 / present as f64 * acc).sqrt()
            }
        }
        // `hamming` and its `matching` alias stay FLOAT (module docs).
        TsneMetric::Hamming => {
            let mut ne = 0usize;
            for k in 0..d {
                if a[k] != b[k] {
                    ne += 1;
                }
            }
            ne as f64 / d.max(1) as f64
        }
        TsneMetric::Precomputed => prep.data[ia * d + ib],
        TsneMetric::WMinkowski => return Err(MetricError::WMinkowskiRemoved),
        // Handled by the `prep.boolean` branch above.
        TsneMetric::Jaccard
        | TsneMetric::Dice
        | TsneMetric::RogersTanimoto
        | TsneMetric::RussellRao
        | TsneMetric::SokalSneath
        | TsneMetric::Yule => unreachable!("boolean metrics take the prepared-bool branch"),
    })
}

/// The six bool-cast metrics, over the classic contingency counts. `a`/`b` hold
/// `0.0`/`1.0` indicators from [`prepare_rows`].
#[inline]
fn boolean_distance(a: &[f64], b: &[f64], metric: TsneMetric) -> Result<f64, MetricError> {
    let (mut ctt, mut ctf, mut cft, mut cff) = (0usize, 0usize, 0usize, 0usize);
    for (&av, &bv) in a.iter().zip(b) {
        match (av != 0.0, bv != 0.0) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            (false, false) => cff += 1,
        }
    }
    let n = a.len() as f64;
    let (ctt, ctf, cft, cff) = (ctt as f64, ctf as f64, cft as f64, cff as f64);
    let disagree = ctf + cft;

    Ok(match metric {
        TsneMetric::Jaccard => {
            let den = ctt + disagree;
            // scipy returns 0 for two all-false rows rather than NaN.
            if den > 0.0 {
                disagree / den
            } else {
                0.0
            }
        }
        TsneMetric::Dice => {
            let den = 2.0 * ctt + disagree;
            // scipy returns NaN here (0/0) and sklearn propagates it; mirrored.
            if den > 0.0 {
                disagree / den
            } else {
                f64::NAN
            }
        }
        TsneMetric::RogersTanimoto => {
            let den = ctt + cff + 2.0 * disagree;
            if den > 0.0 {
                2.0 * disagree / den
            } else {
                0.0
            }
        }
        TsneMetric::RussellRao => (n - ctt) / n.max(1.0),
        TsneMetric::SokalSneath => {
            let den = ctt + 2.0 * disagree;
            // scipy RAISES on this pair; mirrored as a typed error.
            if den > 0.0 {
                2.0 * disagree / den
            } else {
                return Err(MetricError::SokalSneathAllZero);
            }
        }
        TsneMetric::Yule => {
            let den = ctt * cff + ctf * cft;
            if den > 0.0 {
                2.0 * ctf * cft / den
            } else {
                0.0
            }
        }
        _ => unreachable!("boolean_distance is only reached for the six bool metrics"),
    })
}

/// Reject metric/data pairs that cannot be evaluated at all, BEFORE any
/// `O(n²)` work: `haversine` off 2 features, a non-square `precomputed`, and
/// the removed `wminkowski`.
pub fn validate_metric_geometry(
    n: usize,
    d: usize,
    metric: TsneMetric,
) -> Result<(), AlgoError> {
    match metric {
        TsneMetric::Haversine if d != 2 => Err(MetricError::HaversineNot2D.into_algo_error()),
        TsneMetric::Precomputed if n != d => {
            Err(MetricError::PrecomputedNotSquare.into_algo_error())
        }
        TsneMetric::WMinkowski => Err(MetricError::WMinkowskiRemoved.into_algo_error()),
        _ => Ok(()),
    }
}

// ===========================================================================
// Dense pairwise
// ===========================================================================

/// Smallest row block worth its own thread (the `umap_host_knn` precedent).
const MIN_ROWS_PER_THREAD: usize = 8;

/// The dense `n × n` SQUARED distance matrix t-SNE's exact method consumes.
///
/// This is `_fit`'s whole distance stage: evaluate the metric, then square
/// (sklearn's `distances **= 2`, skipped for `euclidean` because it asks for
/// `squared=True` directly — the same value either way). `precomputed` takes
/// `X` as the distance matrix and squares it like any other non-euclidean
/// metric.
///
/// Only the upper triangle is evaluated and mirrored: every supported metric is
/// symmetric, so the lower half is a copy, and skipping it halves the dominant
/// term.
pub fn pairwise_squared(
    x: &[f64],
    n: usize,
    d: usize,
    metric: TsneMetric,
    rp: &ResolvedMetricParams,
    threads: usize,
) -> Result<Vec<f64>, AlgoError> {
    validate_metric_geometry(n, d, metric)?;
    let prep = prepare_rows(x, n, d, metric);
    let mut out = vec![0.0f64; n * n];

    // Rows are independent but their COSTS are not: row `i` evaluates `n − i`
    // pairs under the triangle. A contiguous split would leave the last worker
    // idle, so rows are dealt round-robin by worker id, which balances to
    // within one row without any work-stealing machinery.
    let units = threads.max(1).min(n.div_ceil(MIN_ROWS_PER_THREAD).max(1));
    let err = std::sync::atomic::AtomicUsize::new(usize::MAX);
    {
        let out_ptr = SendPtr(out.as_mut_ptr());
        let prep = &prep;
        let err = &err;
        let run = move |unit: usize| {
            let out_ptr = out_ptr;
            let mut local_err: Option<MetricError> = None;
            let mut i = unit;
            while i < n {
                for j in i..n {
                    let v = if i == j {
                        0.0
                    } else {
                        match pair_distance(prep, d, i, j, metric, rp) {
                            Ok(v) => v,
                            Err(e) => {
                                local_err = Some(e);
                                0.0
                            }
                        }
                    };
                    let sq = v * v;
                    // SAFETY: `(i, j)` and `(j, i)` are written by exactly one
                    // unit — the one that owns row `i` under the round-robin
                    // deal — and no unit reads another's cells.
                    unsafe {
                        *out_ptr.0.add(i * n + j) = sq;
                        *out_ptr.0.add(j * n + i) = sq;
                    }
                }
                i += units;
            }
            if let Some(e) = local_err {
                err.store(e as usize, std::sync::atomic::Ordering::Relaxed);
            }
        };
        if units <= 1 {
            run(0);
        } else {
            std::thread::scope(|scope| {
                for unit in 1..units {
                    let run = &run;
                    scope.spawn(move || run(unit));
                }
                run(0);
            });
        }
    }
    if let Some(e) = decode_metric_error(err.load(std::sync::atomic::Ordering::Relaxed)) {
        return Err(e.into_algo_error());
    }

    // sklearn's `_fit` guard, applied to the metric's own output. A negative
    // SQUARED distance is impossible, so the check reads the pre-square sign
    // via the only metric that can go negative under a user-supplied
    // `precomputed` matrix.
    if metric == TsneMetric::Precomputed && x.iter().any(|&v| v < 0.0) {
        return Err(MetricError::NegativeDistance.into_algo_error());
    }
    Ok(out)
}

/// `MetricError` is a fieldless enum, so a discriminant round-trips through the
/// atomic the worker threads report through.
fn decode_metric_error(code: usize) -> Option<MetricError> {
    Some(match code {
        0 => MetricError::HaversineNot2D,
        1 => MetricError::WMinkowskiRemoved,
        2 => MetricError::SokalSneathAllZero,
        3 => MetricError::BadMetricParamShape,
        4 => MetricError::PrecomputedNotSquare,
        5 => MetricError::NegativeDistance,
        _ => return None,
    })
}

/// A raw pointer the scoped workers may capture. Disjointness is a property of
/// the round-robin row deal above, not of the type.
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
// SAFETY: each unit writes only the cells of the rows it owns (see the deal in
// `pairwise_squared`), and the scope joins every unit before `out` is read.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
