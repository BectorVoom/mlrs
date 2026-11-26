//! Pairwise distances and kernels — sklearn's `sklearn.metrics.pairwise`
//! surface (host engine).
//!
//! Every function takes row-major `rows × d` operands and produces the
//! row-major `rows_x × rows_y` matrix. The metric set is sklearn's
//! `_VALID_METRICS` minus the removed `wminkowski` (its own `cdist` backend,
//! scipy, no longer has it), each implemented directly from its definition in
//! f64:
//!
//! | family | metrics |
//! |---|---|
//! | sklearn-native | `euclidean`/`l2`, `manhattan`/`l1`/`cityblock`, `cosine`, `haversine`, `nan_euclidean` |
//! | scipy real-valued | `sqeuclidean`, `chebyshev`, `minkowski` (+`w`), `seuclidean`, `mahalanobis`, `braycurtis`, `canberra`, `correlation`, `hamming`/`matching` |
//! | scipy boolean (inputs binarized) | `dice`, `jaccard`, `rogerstanimoto`, `russellrao`, `sokalsneath`, `yule` |
//!
//! ## Why direct, not the GEMM expansion
//! sklearn's own `euclidean_distances` computes `‖x‖² + ‖y‖² − 2x·y`, which
//! loses roughly half the significant digits when two rows nearly coincide.
//! The direct `Σ (x − y)²` is never less accurate, so it is used here; the
//! device engine (`crates/mlrs-backend/src/prims/distance.rs`) makes the same
//! choice for its `distance_direct` path.
//!
//! ## Parallelism
//! The `rows_x × rows_y` loop is split over output row blocks on
//! `std::thread::scope` workers — no new dependency, and every worker writes
//! only its own disjoint slice of the output.
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_pairwise_test.rs` (AGENTS.md
//! §2).

use super::MetricError;

/// A distance metric, with its parameters resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum Distance {
    Euclidean,
    SqEuclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    /// `(Σ w_k |x_k − y_k|^p)^(1/p)`; `w = None` is unweighted.
    Minkowski { p: f64, w: Option<Vec<f64>> },
    /// Euclidean with each coordinate scaled by `1 / V_k`.
    SEuclidean { v: Vec<f64> },
    /// `sqrt((x − y)ᵀ VI (x − y))`; `vi` is row-major `d × d`.
    Mahalanobis { vi: Vec<f64> },
    BrayCurtis,
    Canberra,
    Correlation,
    /// Fraction of coordinates that differ (scipy `hamming`, also `matching`).
    Hamming,
    /// Great-circle distance on the unit sphere; `d` must be 2 (lat, lon).
    Haversine,
    /// Euclidean over the coordinates present in both rows, rescaled by
    /// `d / n_present`; `missing` marks an absent value (NaN by default).
    NanEuclidean { missing: f64 },
    Dice,
    Jaccard,
    RogersTanimoto,
    RussellRao,
    SokalSneath,
    Yule,
}

impl Distance {
    /// Parse a metric NAME that needs no parameter (the parametrized ones are
    /// built by the caller). `Err(UnknownMetric)` otherwise.
    pub fn from_name(name: &str) -> Result<Distance, MetricError> {
        Ok(match name {
            "euclidean" | "l2" => Distance::Euclidean,
            "sqeuclidean" => Distance::SqEuclidean,
            "manhattan" | "l1" | "cityblock" => Distance::Manhattan,
            "cosine" => Distance::Cosine,
            "chebyshev" => Distance::Chebyshev,
            "braycurtis" => Distance::BrayCurtis,
            "canberra" => Distance::Canberra,
            "correlation" => Distance::Correlation,
            "hamming" | "matching" => Distance::Hamming,
            "haversine" => Distance::Haversine,
            "dice" => Distance::Dice,
            "jaccard" => Distance::Jaccard,
            "rogerstanimoto" => Distance::RogersTanimoto,
            "russellrao" => Distance::RussellRao,
            "sokalsneath" => Distance::SokalSneath,
            "yule" => Distance::Yule,
            _ => return Err(MetricError::UnknownMetric),
        })
    }

    fn is_boolean(&self) -> bool {
        matches!(
            self,
            Distance::Dice
                | Distance::Jaccard
                | Distance::RogersTanimoto
                | Distance::RussellRao
                | Distance::SokalSneath
                | Distance::Yule
        )
    }
}

/// A pairwise kernel, with its parameters resolved (sklearn's `gamma=None`
/// default of `1 / n_features` is resolved by the caller).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    Linear,
    Poly { gamma: f64, degree: f64, coef0: f64 },
    Rbf { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
    Laplacian { gamma: f64 },
    Cosine,
    Chi2 { gamma: f64 },
    AdditiveChi2,
}

/// Validate one row-major operand and return its row count.
fn rows_of(a: &[f64], d: usize) -> Result<usize, MetricError> {
    if d == 0 || a.len() % d != 0 {
        return Err(MetricError::BadShape);
    }
    Ok(a.len() / d)
}

/// Fill the row-major `rx × ry` output by calling `cell(i, j)` for every
/// entry, split over output-row blocks on scoped threads.
fn par_fill(rx: usize, ry: usize, cell: impl Fn(usize, usize) -> f64 + Sync) -> Vec<f64> {
    let mut out = vec![0.0f64; rx * ry];
    if ry == 0 || rx == 0 {
        return out;
    }
    let work = rx * ry;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rx)
        .min(work / 4096 + 1);
    if threads <= 1 {
        for i in 0..rx {
            for j in 0..ry {
                out[i * ry + j] = cell(i, j);
            }
        }
        return out;
    }
    let rows_per = rx.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(rows_per * ry).enumerate() {
            let cell = &cell;
            s.spawn(move || {
                let r0 = t * rows_per;
                // Row-at-a-time: no per-cell div/mod to recover (i, j).
                for (r, row) in chunk.chunks_mut(ry).enumerate() {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = cell(r0 + r, j);
                    }
                }
            });
        }
    });
    out
}

/// `Σ f(u_k, v_k)` with four independent accumulators. A single running sum
/// is a serial dependency chain the compiler may not reorder (f64 addition is
/// not associative), so it neither pipelines nor vectorizes; four lanes let it
/// do both. Measured on the 3000 × 2000 × 50 euclidean case this is what
/// brought the host engine level with sklearn's BLAS path.
#[inline]
fn sum4(u: &[f64], v: &[f64], f: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = [0.0f64; 4];
    let (uc, vc) = (u.chunks_exact(4), v.chunks_exact(4));
    let (ur, vr) = (uc.remainder(), vc.remainder());
    for (a, b) in uc.zip(vc) {
        // Fixed-size chunks: no bounds checks inside, so the four lanes map
        // onto SIMD registers.
        acc[0] += f(a[0], b[0]);
        acc[1] += f(a[1], b[1]);
        acc[2] += f(a[2], b[2]);
        acc[3] += f(a[3], b[3]);
    }
    let tail: f64 = ur.iter().zip(vr).map(|(&a, &b)| f(a, b)).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[inline]
fn sq_dist(u: &[f64], v: &[f64]) -> f64 {
    sum4(u, v, |a, b| (a - b) * (a - b))
}

#[inline]
fn dot(u: &[f64], v: &[f64]) -> f64 {
    sum4(u, v, |a, b| a * b)
}

#[inline]
fn l1(u: &[f64], v: &[f64]) -> f64 {
    sum4(u, v, |a, b| (a - b).abs())
}

fn norms(a: &[f64], d: usize) -> Vec<f64> {
    a.chunks(d).map(|r| dot(r, r).sqrt()).collect()
}

/// Boolean contingency counts `(ctt, ctf, cft, cff)` of two binarized rows.
fn bool_counts(u: &[f64], v: &[f64]) -> (f64, f64, f64, f64) {
    let (mut tt, mut tf, mut ft, mut ff) = (0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in u.iter().zip(v) {
        match (a != 0.0, b != 0.0) {
            (true, true) => tt += 1.0,
            (true, false) => tf += 1.0,
            (false, true) => ft += 1.0,
            (false, false) => ff += 1.0,
        }
    }
    (tt, tf, ft, ff)
}

/// The full `rows_x × rows_y` distance matrix. `y = None` means `Y = X`, for
/// which the diagonal is set to exactly 0 (sklearn zero-fills it for its own
/// metrics, and scipy's `pdist`/`squareform` route produces exact zeros) —
/// except a `nan_euclidean` row with no present coordinate, whose diagonal is
/// NaN in sklearn too.
pub fn pairwise_distances(
    x: &[f64],
    y: Option<&[f64]>,
    d: usize,
    metric: &Distance,
) -> Result<Vec<f64>, MetricError> {
    let rx = rows_of(x, d)?;
    let same = y.is_none();
    let y = y.unwrap_or(x);
    let ry = rows_of(y, d)?;
    match metric {
        Distance::Haversine if d != 2 => return Err(MetricError::BadShape),
        Distance::Minkowski { p, w } => {
            if !(*p > 0.0) || w.as_ref().is_some_and(|w| w.len() != d || w.iter().any(|&v| v < 0.0)) {
                return Err(MetricError::InvalidParameter);
            }
        }
        Distance::SEuclidean { v } if v.len() != d => return Err(MetricError::BadShape),
        Distance::Mahalanobis { vi } if vi.len() != d * d => return Err(MetricError::BadShape),
        _ => {}
    }
    let xr = |i: usize| &x[i * d..(i + 1) * d];
    let yr = |j: usize| &y[j * d..(j + 1) * d];

    // Per-row precomputations for the metrics that normalize.
    let (xn, yn) = match metric {
        Distance::Cosine => (norms(x, d), norms(y, d)),
        _ => (Vec::new(), Vec::new()),
    };
    let centered = |a: &[f64]| -> Vec<f64> {
        a.chunks(d)
            .flat_map(|r| {
                let m = r.iter().sum::<f64>() / d as f64;
                r.iter().map(move |v| v - m).collect::<Vec<_>>()
            })
            .collect()
    };
    let (xc, yc) = match metric {
        Distance::Correlation => (centered(x), centered(y)),
        _ => (Vec::new(), Vec::new()),
    };
    let (xcn, ycn) = match metric {
        Distance::Correlation => (norms(&xc, d), norms(&yc, d)),
        _ => (Vec::new(), Vec::new()),
    };

    // The hot metrics get their own monomorphized fill (the per-cell `match`
    // below blocks inlining the reduction into the loop).
    let fast = match metric {
        Distance::Euclidean => Some(par_fill(rx, ry, |i, j| sq_dist(xr(i), yr(j)).sqrt())),
        Distance::SqEuclidean => Some(par_fill(rx, ry, |i, j| sq_dist(xr(i), yr(j)))),
        Distance::Manhattan => Some(par_fill(rx, ry, |i, j| l1(xr(i), yr(j)))),
        Distance::Cosine => Some(par_fill(rx, ry, |i, j| {
            let nx = if xn[i] == 0.0 { 1.0 } else { xn[i] };
            let ny = if yn[j] == 0.0 { 1.0 } else { yn[j] };
            (1.0 - dot(xr(i), yr(j)) / (nx * ny)).clamp(0.0, 2.0)
        })),
        _ => None,
    };
    let mut out = fast.unwrap_or_else(|| par_fill(rx, ry, |i, j| {
        let (u, v) = (xr(i), yr(j));
        match metric {
            Distance::Euclidean => sq_dist(u, v).sqrt(),
            Distance::SqEuclidean => sq_dist(u, v),
            Distance::Manhattan => l1(u, v),
            Distance::Chebyshev => u.iter().zip(v).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
            Distance::Minkowski { p, w } => {
                let s: f64 = match w {
                    None => u.iter().zip(v).map(|(a, b)| (a - b).abs().powf(*p)).sum(),
                    Some(w) => u
                        .iter()
                        .zip(v)
                        .zip(w)
                        .map(|((a, b), wk)| wk * (a - b).abs().powf(*p))
                        .sum(),
                };
                s.powf(1.0 / p)
            }
            Distance::SEuclidean { v: var } => u
                .iter()
                .zip(v)
                .zip(var)
                .map(|((a, b), s)| (a - b) * (a - b) / s)
                .sum::<f64>()
                .sqrt(),
            Distance::Mahalanobis { vi } => {
                let diff: Vec<f64> = u.iter().zip(v).map(|(a, b)| a - b).collect();
                let mut q = 0.0f64;
                for r in 0..d {
                    let mut acc = 0.0f64;
                    for c in 0..d {
                        acc += vi[r * d + c] * diff[c];
                    }
                    q += diff[r] * acc;
                }
                q.sqrt()
            }
            Distance::Cosine => {
                // sklearn: 1 - cos_sim, clipped to [0, 2]; a zero row is left
                // unnormalized (similarity 0, distance 1).
                let nx = if xn[i] == 0.0 { 1.0 } else { xn[i] };
                let ny = if yn[j] == 0.0 { 1.0 } else { yn[j] };
                (1.0 - dot(u, v) / (nx * ny)).clamp(0.0, 2.0)
            }
            Distance::Correlation => {
                let (cu, cv) = (&xc[i * d..(i + 1) * d], &yc[j * d..(j + 1) * d]);
                let dist = 1.0 - dot(cu, cv) / (xcn[i] * ycn[j]);
                dist.clamp(0.0, 2.0)
            }
            Distance::BrayCurtis => {
                let num: f64 = u.iter().zip(v).map(|(a, b)| (a - b).abs()).sum();
                let den: f64 = u.iter().zip(v).map(|(a, b)| (a + b).abs()).sum();
                num / den
            }
            Distance::Canberra => u
                .iter()
                .zip(v)
                .map(|(a, b)| {
                    let den = a.abs() + b.abs();
                    if den == 0.0 { 0.0 } else { (a - b).abs() / den }
                })
                .sum(),
            Distance::Hamming => {
                u.iter().zip(v).filter(|(a, b)| a != b).count() as f64 / d as f64
            }
            Distance::Haversine => {
                let s1 = ((v[0] - u[0]) / 2.0).sin();
                let s2 = ((v[1] - u[1]) / 2.0).sin();
                2.0 * (s1 * s1 + u[0].cos() * v[0].cos() * s2 * s2).sqrt().asin()
            }
            Distance::NanEuclidean { missing } => {
                let is_missing = |a: f64| if missing.is_nan() { a.is_nan() } else { a == *missing };
                let mut sq = 0.0f64;
                let mut present = 0usize;
                for (&a, &b) in u.iter().zip(v) {
                    if !is_missing(a) && !is_missing(b) {
                        sq += (a - b) * (a - b);
                        present += 1;
                    }
                }
                if present == 0 {
                    f64::NAN
                } else {
                    (d as f64 / present as f64 * sq).sqrt()
                }
            }
            m if m.is_boolean() => {
                let (tt, tf, ft, ff) = bool_counts(u, v);
                let n = d as f64;
                match m {
                    Distance::Dice => {
                        let den = 2.0 * tt + tf + ft;
                        if den == 0.0 { 0.0 } else { (tf + ft) / den }
                    }
                    Distance::Jaccard => {
                        let den = tt + tf + ft;
                        if den == 0.0 { 0.0 } else { (tf + ft) / den }
                    }
                    Distance::RogersTanimoto => {
                        let r = 2.0 * (tf + ft);
                        r / (tt + ff + r)
                    }
                    Distance::RussellRao => (n - tt) / n,
                    Distance::SokalSneath => {
                        let r = 2.0 * (tf + ft);
                        r / (tt + r)
                    }
                    Distance::Yule => {
                        let half_r = tf * ft;
                        if half_r == 0.0 { 0.0 } else { 2.0 * half_r / (tt * ff + half_r) }
                    }
                    _ => unreachable!("is_boolean covers exactly these six"),
                }
            }
            _ => unreachable!("every Distance variant is handled above"),
        }
    }));
    if same {
        // sklearn's nan_euclidean zero-fills the diagonal and THEN marks every
        // pair with no shared present coordinate as NaN, so an all-missing row
        // keeps a NaN diagonal; every other metric gets an exact 0.
        let keep_nan = matches!(metric, Distance::NanEuclidean { .. });
        for i in 0..rx {
            let cell = &mut out[i * ry + i];
            if !(keep_nan && cell.is_nan()) {
                *cell = 0.0;
            }
        }
    }
    Ok(out)
}

/// The full `rows_x × rows_y` kernel matrix (`y = None` means `Y = X`).
/// `AdditiveChi2`/`Chi2` require non-negative input (`Err(DomainError)`).
pub fn pairwise_kernels(x: &[f64], y: Option<&[f64]>, d: usize, kernel: Kernel) -> Result<Vec<f64>, MetricError> {
    let rx = rows_of(x, d)?;
    let y = y.unwrap_or(x);
    let ry = rows_of(y, d)?;
    if matches!(kernel, Kernel::AdditiveChi2 | Kernel::Chi2 { .. }) && x.iter().chain(y).any(|&v| v < 0.0) {
        return Err(MetricError::DomainError);
    }
    let (xn, yn) = match kernel {
        Kernel::Cosine => (norms(x, d), norms(y, d)),
        _ => (Vec::new(), Vec::new()),
    };
    let xr = |i: usize| &x[i * d..(i + 1) * d];
    let yr = |j: usize| &y[j * d..(j + 1) * d];
    let chi = |u: &[f64], v: &[f64]| -> f64 {
        -u.iter()
            .zip(v)
            .map(|(a, b)| {
                let s = a + b;
                if s == 0.0 { 0.0 } else { (a - b) * (a - b) / s }
            })
            .sum::<f64>()
    };
    Ok(par_fill(rx, ry, |i, j| {
        let (u, v) = (xr(i), yr(j));
        match kernel {
            Kernel::Linear => dot(u, v),
            Kernel::Poly { gamma, degree, coef0 } => (gamma * dot(u, v) + coef0).powf(degree),
            Kernel::Sigmoid { gamma, coef0 } => (gamma * dot(u, v) + coef0).tanh(),
            Kernel::Rbf { gamma } => (-gamma * sq_dist(u, v)).exp(),
            Kernel::Laplacian { gamma } => (-gamma * l1(u, v)).exp(),
            Kernel::Cosine => {
                let nx = if xn[i] == 0.0 { 1.0 } else { xn[i] };
                let ny = if yn[j] == 0.0 { 1.0 } else { yn[j] };
                dot(u, v) / (nx * ny)
            }
            Kernel::AdditiveChi2 => chi(u, v),
            Kernel::Chi2 { gamma } => (gamma * chi(u, v)).exp(),
        }
    }))
}

/// Row-wise ("paired") distances between `x[i]` and `y[i]`: `euclidean`,
/// `manhattan`, or sklearn's `paired_cosine_distances` (`0.5·‖x̂ − ŷ‖²`), or any
/// other [`Distance`] evaluated row-against-row.
pub fn paired_distances(x: &[f64], y: &[f64], d: usize, metric: &Distance) -> Result<Vec<f64>, MetricError> {
    let rx = rows_of(x, d)?;
    if y.len() != x.len() {
        return Err(MetricError::LengthMismatch);
    }
    Ok((0..rx)
        .map(|i| {
            let (u, v) = (&x[i * d..(i + 1) * d], &y[i * d..(i + 1) * d]);
            match metric {
                Distance::Cosine => {
                    let nu = u.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let nu = if nu == 0.0 { 1.0 } else { nu };
                    let nv = if nv == 0.0 { 1.0 } else { nv };
                    0.5 * u.iter().zip(v).map(|(a, b)| (a / nu - b / nv).powi(2)).sum::<f64>()
                }
                m => pairwise_distances(u, Some(v), d, m).map(|r| r[0]).unwrap_or(f64::NAN),
            }
        })
        .collect())
}

/// `pairwise_distances_argmin_min(X, Y, axis=1)`: for every row of `x`, the
/// index of its nearest `y` row (the FIRST one on a tie) and that distance.
/// Streams one `x` row at a time, so the `rows_x × rows_y` matrix is never
/// materialized.
pub fn argmin_min(x: &[f64], y: &[f64], d: usize, metric: &Distance) -> Result<(Vec<usize>, Vec<f64>), MetricError> {
    let rx = rows_of(x, d)?;
    let ry = rows_of(y, d)?;
    if ry == 0 {
        return Err(MetricError::EmptyInput);
    }
    let block = 256usize;
    let mut idx = Vec::with_capacity(rx);
    let mut val = Vec::with_capacity(rx);
    for start in (0..rx).step_by(block) {
        let stop = (start + block).min(rx);
        let dists = pairwise_distances(&x[start * d..stop * d], Some(y), d, metric)?;
        for r in 0..stop - start {
            let row = &dists[r * ry..(r + 1) * ry];
            let (mut bi, mut bv) = (0usize, f64::INFINITY);
            for (j, &v) in row.iter().enumerate() {
                if v < bv {
                    bv = v;
                    bi = j;
                }
            }
            if bv == f64::INFINITY {
                bv = row[0];
            }
            idx.push(bi);
            val.push(bv);
        }
    }
    Ok((idx, val))
}

/// Solve `A · X = I` by Gauss–Jordan with partial pivoting (the default
/// `mahalanobis` `VI = inv(cov)`). `Err(DomainError)` for a singular matrix.
pub fn invert(a: &[f64], d: usize) -> Result<Vec<f64>, MetricError> {
    if a.len() != d * d {
        return Err(MetricError::BadShape);
    }
    let mut m = a.to_vec();
    let mut inv = vec![0.0f64; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&p, &q| m[p * d + col].abs().total_cmp(&m[q * d + col].abs()))
            .expect("non-empty pivot range");
        if m[piv * d + col] == 0.0 {
            return Err(MetricError::DomainError);
        }
        for k in 0..d {
            m.swap(col * d + k, piv * d + k);
            inv.swap(col * d + k, piv * d + k);
        }
        let p = m[col * d + col];
        for k in 0..d {
            m[col * d + k] /= p;
            inv[col * d + k] /= p;
        }
        for r in 0..d {
            if r != col {
                let f = m[r * d + col];
                if f != 0.0 {
                    for k in 0..d {
                        m[r * d + k] -= f * m[col * d + k];
                        inv[r * d + k] -= f * inv[col * d + k];
                    }
                }
            }
        }
    }
    Ok(inv)
}

/// Column variances (`ddof = 1`) and covariance matrix of a row-major `n × d`
/// matrix — the `seuclidean`/`mahalanobis` defaults sklearn derives from `X`.
pub fn column_var_cov(x: &[f64], d: usize) -> Result<(Vec<f64>, Vec<f64>), MetricError> {
    let n = rows_of(x, d)?;
    if n < 2 {
        return Err(MetricError::TooFewPoints);
    }
    let mut mean = vec![0.0f64; d];
    for r in x.chunks(d) {
        for k in 0..d {
            mean[k] += r[k];
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    let mut cov = vec![0.0f64; d * d];
    for r in x.chunks(d) {
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] += (r[a] - mean[a]) * (r[b] - mean[b]);
            }
        }
    }
    cov.iter_mut().for_each(|c| *c /= (n - 1) as f64);
    let var = (0..d).map(|k| cov[k * d + k]).collect();
    Ok((var, cov))
}
