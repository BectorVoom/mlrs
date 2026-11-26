//! Clustering metrics — sklearn's `sklearn.metrics.cluster` surface.
//!
//! The SUPERVISED scores (mutual-information family, homogeneity /
//! completeness / V-measure, Rand and adjusted Rand, Fowlkes–Mallows, the pair
//! confusion matrix) are all functions of one sparse contingency table, built
//! here in a single O(n log n) pass ([`Contingency::new`]) and stored as its
//! nonzero cells plus the two marginals — exactly the `csr` form sklearn uses,
//! so a 10⁵-cluster labelling never allocates a dense 10⁵ × 10⁵ table.
//!
//! The UNSUPERVISED scores (`silhouette_samples`, `calinski_harabasz_score`,
//! `davies_bouldin_score`) take the row-major feature matrix. Silhouette
//! consumes the pairwise distances one row block at a time
//! ([`silhouette_samples_from_distances`] is the per-block reduction), so the
//! full `n × n` matrix never has to exist.
//!
//! Integer arithmetic is kept exact where sklearn's is: pair counts are `i128`
//! (an `n = 10⁶` labelling has ~10¹² pairs and products of two of them
//! overflow `i64`), and only the final ratios become `f64`.
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_cluster_test.rs` (AGENTS.md
//! §2).

use super::MetricError;

/// A sparse contingency table: `cells` holds `(class, cluster, count)` for the
/// nonzero entries; `row`/`col` are the class/cluster marginals.
#[derive(Debug, Clone, PartialEq)]
pub struct Contingency {
    pub n_samples: u64,
    pub cells: Vec<(u32, u32, u64)>,
    pub row: Vec<u64>,
    pub col: Vec<u64>,
}

/// Encode labels to dense `0..K` codes in sorted-label order.
fn encode(labels: &[i64]) -> (Vec<u32>, usize) {
    let mut uniq = labels.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let codes = labels
        .iter()
        .map(|v| uniq.binary_search(v).expect("label is in its own unique set") as u32)
        .collect();
    (codes, uniq.len())
}

impl Contingency {
    /// Build the table from two equal-length labellings.
    pub fn new(labels_true: &[i64], labels_pred: &[i64]) -> Result<Self, MetricError> {
        if labels_true.len() != labels_pred.len() {
            return Err(MetricError::LengthMismatch);
        }
        let (ct, nc) = encode(labels_true);
        let (cp, nk) = encode(labels_pred);
        let mut keys: Vec<u64> = ct
            .iter()
            .zip(&cp)
            .map(|(&a, &b)| (u64::from(a) << 32) | u64::from(b))
            .collect();
        keys.sort_unstable();
        let mut cells = Vec::new();
        let mut i = 0;
        while i < keys.len() {
            let mut j = i;
            while j < keys.len() && keys[j] == keys[i] {
                j += 1;
            }
            cells.push(((keys[i] >> 32) as u32, keys[i] as u32, (j - i) as u64));
            i = j;
        }
        let mut row = vec![0u64; nc];
        let mut col = vec![0u64; nk];
        for &(a, b, c) in &cells {
            row[a as usize] += c;
            col[b as usize] += c;
        }
        Ok(Contingency {
            n_samples: labels_true.len() as u64,
            cells,
            row,
            col,
        })
    }

    /// The dense `n_classes × n_clusters` table, row-major (sklearn's
    /// `contingency_matrix(sparse=False)`).
    pub fn dense(&self) -> Vec<u64> {
        let nk = self.col.len();
        let mut out = vec![0u64; self.row.len() * nk];
        for &(a, b, c) in &self.cells {
            out[a as usize * nk + b as usize] = c;
        }
        out
    }
}

/// sklearn's `contingency_matrix` (dense), row-major, with its shape.
pub fn contingency_matrix(
    labels_true: &[i64],
    labels_pred: &[i64],
) -> Result<(Vec<u64>, usize, usize), MetricError> {
    let c = Contingency::new(labels_true, labels_pred)?;
    Ok((c.dense(), c.row.len(), c.col.len()))
}

/// Entropy (nats) of a labelling's class counts; 1.0 for an empty labelling
/// and 0.0 for a single class, as sklearn's `_entropy`.
fn entropy_of_counts(counts: &[u64], n: u64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if counts.len() == 1 {
        return 0.0;
    }
    let total = n as f64;
    let log_total = total.ln();
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let c = c as f64;
            c / total * (c.ln() - log_total)
        })
        .sum::<f64>()
}

/// `entropy(labels)` (sklearn's `sklearn.metrics.cluster.entropy`).
pub fn entropy(labels: &[i64]) -> f64 {
    let (codes, k) = encode(labels);
    let mut counts = vec![0u64; k];
    for c in codes {
        counts[c as usize] += 1;
    }
    entropy_of_counts(&counts, labels.len() as u64)
}

/// Mutual information (nats) of a contingency table, term for term as
/// sklearn's `mutual_info_score`: each cell's term is zeroed below f64 epsilon
/// in magnitude and the sum is clipped at 0.
pub fn mutual_info_from_contingency(c: &Contingency) -> f64 {
    if c.row.len() == 1 || c.col.len() == 1 {
        return 0.0;
    }
    let total: f64 = c.cells.iter().map(|&(_, _, v)| v as f64).sum();
    let log_total = total.ln();
    let row_sum: f64 = c.row.iter().map(|&v| v as f64).sum();
    let col_sum: f64 = c.col.iter().map(|&v| v as f64).sum();
    let mut mi = 0.0f64;
    for &(a, b, v) in &c.cells {
        let nm = v as f64 / total;
        let outer = (c.row[a as usize] as i64 as f64) * (c.col[b as usize] as i64 as f64);
        let log_outer = -outer.ln() + row_sum.ln() + col_sum.ln();
        let term = nm * ((v as f64).ln() - log_total) + nm * log_outer;
        if term.abs() >= f64::EPSILON {
            mi += term;
        }
    }
    mi.max(0.0)
}

/// `mutual_info_score(labels_true, labels_pred)`.
pub fn mutual_info_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, MetricError> {
    Ok(mutual_info_from_contingency(&Contingency::new(labels_true, labels_pred)?))
}

impl Contingency {
    /// Rebuild the sparse table from a dense row-major `rows × cols`
    /// non-negative count matrix (sklearn's `contingency=` argument). Counts
    /// are truncated to integers, as sklearn's `dtype=int` check implies.
    pub fn from_dense(counts: &[f64], rows: usize, cols: usize) -> Result<Self, MetricError> {
        if rows * cols != counts.len() {
            return Err(MetricError::BadShape);
        }
        if counts.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(MetricError::DomainError);
        }
        let mut cells = Vec::new();
        let mut row = vec![0u64; rows];
        let mut col = vec![0u64; cols];
        for a in 0..rows {
            for b in 0..cols {
                let v = counts[a * cols + b] as u64;
                if v > 0 {
                    cells.push((a as u32, b as u32, v));
                    row[a] += v;
                    col[b] += v;
                }
            }
        }
        let n_samples = row.iter().sum();
        Ok(Contingency {
            n_samples,
            cells,
            row,
            col,
        })
    }
}

/// `mutual_info_score(None, None, contingency=C)` for a dense row-major
/// `rows × cols` non-negative count table.
pub fn mutual_info_score_contingency(counts: &[f64], rows: usize, cols: usize) -> Result<f64, MetricError> {
    Ok(mutual_info_from_contingency(&Contingency::from_dense(counts, rows, cols)?))
}

/// `expected_mutual_information(contingency, n_samples)` for a dense count
/// table; `n_samples` is sklearn's explicit argument (normally the table's
/// total).
pub fn expected_mutual_information_dense(
    counts: &[f64],
    rows: usize,
    cols: usize,
    n_samples: u64,
) -> Result<f64, MetricError> {
    let mut c = Contingency::from_dense(counts, rows, cols)?;
    if c.row.iter().chain(&c.col).any(|&v| v > n_samples) {
        return Err(MetricError::DomainError);
    }
    c.n_samples = n_samples;
    Ok(expected_mutual_information(&c))
}

/// `ln(k!)` for `k = 0..=n`, accumulated exactly from `ln(i)`. The EMI sum
/// below cancels large log-factorials against each other, and an exact table
/// keeps that cancellation at f64 rounding rather than at a series
/// approximation's error.
fn log_factorials(n: u64) -> Vec<f64> {
    let mut t = Vec::with_capacity(n as usize + 1);
    t.push(0.0);
    let mut acc = 0.0f64;
    for i in 1..=n {
        acc += (i as f64).ln();
        t.push(acc);
    }
    t
}

/// Expected mutual information under the hypergeometric permutation model —
/// sklearn's `_expected_mutual_info_fast.expected_mutual_information`, the
/// same triple loop over `(class, cluster, n_ij)`.
pub fn expected_mutual_information(c: &Contingency) -> f64 {
    let a = &c.row;
    let b = &c.col;
    if a.len() == 1 || b.len() == 1 {
        return 0.0;
    }
    let n = c.n_samples;
    let nf = n as f64;
    let lf = log_factorials(n);
    let mut emi = 0.0f64;
    for &ai in a {
        for &bj in b {
            let start = 1.max((ai + bj).saturating_sub(n));
            let end = ai.min(bj);
            let base = lf[ai as usize] + lf[bj as usize] + lf[(n - ai) as usize] + lf[(n - bj) as usize]
                - lf[n as usize];
            for nij in start..=end {
                let nijf = nij as f64;
                let term1 = nijf / nf;
                let term2 = (nf.ln() + nijf.ln()) - (ai as f64).ln() - (bj as f64).ln();
                let gln = base
                    - lf[nij as usize]
                    - lf[(ai - nij) as usize]
                    - lf[(bj - nij) as usize]
                    - lf[(n - ai - bj + nij) as usize];
                emi += term1 * term2 * gln.exp();
            }
        }
    }
    emi
}

/// The `average_method` of the normalized MI scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AverageMethod {
    Min,
    Geometric,
    Arithmetic,
    Max,
}

fn generalized_average(u: f64, v: f64, m: AverageMethod) -> f64 {
    match m {
        AverageMethod::Min => u.min(v),
        AverageMethod::Geometric => (u * v).sqrt(),
        AverageMethod::Arithmetic => (u + v) / 2.0,
        AverageMethod::Max => u.max(v),
    }
}

/// `adjusted_mutual_info_score`: `(MI - E[MI]) / (avg(H_true, H_pred) -
/// E[MI])`, with sklearn's two `±eps` clamps on numerator and denominator.
pub fn adjusted_mutual_info_score(
    labels_true: &[i64],
    labels_pred: &[i64],
    method: AverageMethod,
) -> Result<f64, MetricError> {
    let c = Contingency::new(labels_true, labels_pred)?;
    let (nc, nk) = (c.row.len(), c.col.len());
    if (nc == 1 && nk == 1) || (nc == 0 && nk == 0) {
        return Ok(1.0);
    }
    if nc == 1 || nk == 1 {
        return Ok(0.0);
    }
    let mi = mutual_info_from_contingency(&c);
    let emi = expected_mutual_information(&c);
    let h_true = entropy_of_counts(&c.row, c.n_samples);
    let h_pred = entropy_of_counts(&c.col, c.n_samples);
    let normalizer = generalized_average(h_true, h_pred, method);
    let eps = f64::EPSILON;
    let mut den = normalizer - emi;
    den = if den < 0.0 { den.min(-eps) } else { den.max(eps) };
    let mut num = mi - emi;
    num = if num < 0.0 { num.min(-eps) } else { num.max(eps) };
    Ok(num / den)
}

/// `normalized_mutual_info_score`: `MI / avg(H_true, H_pred)`.
pub fn normalized_mutual_info_score(
    labels_true: &[i64],
    labels_pred: &[i64],
    method: AverageMethod,
) -> Result<f64, MetricError> {
    let c = Contingency::new(labels_true, labels_pred)?;
    let (nc, nk) = (c.row.len(), c.col.len());
    if (nc == 1 && nk == 1) || (nc == 0 && nk == 0) {
        return Ok(1.0);
    }
    let mi = mutual_info_from_contingency(&c);
    if mi == 0.0 {
        return Ok(0.0);
    }
    let h_true = entropy_of_counts(&c.row, c.n_samples);
    let h_pred = entropy_of_counts(&c.col, c.n_samples);
    Ok(mi / generalized_average(h_true, h_pred, method))
}

/// `homogeneity_completeness_v_measure`: `(h, c, v)` with `v` the
/// `beta`-weighted harmonic mean (`beta > 1` favours completeness).
pub fn homogeneity_completeness_v_measure(
    labels_true: &[i64],
    labels_pred: &[i64],
    beta: f64,
) -> Result<(f64, f64, f64), MetricError> {
    if labels_true.is_empty() && labels_pred.is_empty() {
        return Ok((1.0, 1.0, 1.0));
    }
    let c = Contingency::new(labels_true, labels_pred)?;
    let h_c = entropy_of_counts(&c.row, c.n_samples);
    let h_k = entropy_of_counts(&c.col, c.n_samples);
    let mi = mutual_info_from_contingency(&c);
    let h = if h_c != 0.0 { mi / h_c } else { 1.0 };
    let comp = if h_k != 0.0 { mi / h_k } else { 1.0 };
    let v = if h + comp == 0.0 {
        0.0
    } else {
        (1.0 + beta) * h * comp / (beta * h + comp)
    };
    Ok((h, comp, v))
}

/// `fowlkes_mallows_score`: `TP / sqrt((TP + FP)(TP + FN))` over sample pairs.
pub fn fowlkes_mallows_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, MetricError> {
    let c = Contingency::new(labels_true, labels_pred)?;
    let n = c.n_samples as i128;
    let sq = |v: &[u64]| v.iter().map(|&x| (x as i128) * (x as i128)).sum::<i128>();
    let tk: i128 = c.cells.iter().map(|&(_, _, v)| (v as i128) * (v as i128)).sum::<i128>() - n;
    let pk = sq(&c.col) - n;
    let qk = sq(&c.row) - n;
    if tk == 0 {
        return Ok(0.0);
    }
    Ok((tk as f64 / pk as f64).sqrt() * (tk as f64 / qk as f64).sqrt())
}

/// `pair_confusion_matrix`: the `2 × 2` matrix `[[tn, fp], [fn, tp]]` over
/// ORDERED sample pairs (so every entry is even), row-major.
pub fn pair_confusion_matrix(labels_true: &[i64], labels_pred: &[i64]) -> Result<[i128; 4], MetricError> {
    let c = Contingency::new(labels_true, labels_pred)?;
    Ok(pair_confusion_from(&c))
}

fn pair_confusion_from(c: &Contingency) -> [i128; 4] {
    let n = c.n_samples as i128;
    let sum_squares: i128 = c.cells.iter().map(|&(_, _, v)| (v as i128) * (v as i128)).sum();
    // contingency · n_k summed: Σ_cells v · col[b]; and the transpose.
    let c_nk: i128 = c
        .cells
        .iter()
        .map(|&(_, b, v)| v as i128 * c.col[b as usize] as i128)
        .sum();
    let ct_nc: i128 = c
        .cells
        .iter()
        .map(|&(a, _, v)| v as i128 * c.row[a as usize] as i128)
        .sum();
    let tp = sum_squares - n;
    let fp = c_nk - sum_squares;
    let fnn = ct_nc - sum_squares;
    let tn = n * n - fp - fnn - sum_squares;
    [tn, fp, fnn, tp]
}

/// `rand_score`: the fraction of agreeing pairs (1.0 for the degenerate
/// all-agree / no-pair cases).
pub fn rand_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, MetricError> {
    let [tn, fp, fnn, tp] = pair_confusion_matrix(labels_true, labels_pred)?;
    let num = tn + tp;
    let den = tn + fp + fnn + tp;
    if num == den || den == 0 {
        return Ok(1.0);
    }
    Ok(num as f64 / den as f64)
}

/// `adjusted_rand_score`: the Rand index adjusted for chance,
/// `2(tp·tn - fn·fp) / ((tp+fn)(fn+tn) + (tp+fp)(fp+tn))`, computed in `i128`
/// (sklearn uses Python's unbounded ints for exactly this product).
pub fn adjusted_rand_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64, MetricError> {
    let [tn, fp, fnn, tp] = pair_confusion_matrix(labels_true, labels_pred)?;
    if fnn == 0 && fp == 0 {
        return Ok(1.0);
    }
    let num = 2 * (tp * tn - fnn * fp);
    let den = (tp + fnn) * (fnn + tn) + (tp + fp) * (fp + tn);
    Ok(num as f64 / den as f64)
}

/// Encode labels to `0..K` and require `2 <= K <= n - 1` (sklearn's
/// `check_number_of_labels`).
fn encode_checked(labels: &[i64]) -> Result<(Vec<u32>, usize), MetricError> {
    let (codes, k) = encode(labels);
    if !(1 < k && k < labels.len()) {
        return Err(MetricError::TooFewPoints);
    }
    Ok((codes, k))
}

/// The per-block silhouette reduction over a row-major `rows × n` block of
/// distances whose rows are samples `start..start+rows`: returns each row's
/// `(intra, inter)` — the summed distance to its own cluster, and the smallest
/// MEAN distance to another cluster.
pub fn silhouette_block(
    block: &[f64],
    rows: usize,
    start: usize,
    codes: &[u32],
    freqs: &[u64],
) -> Vec<(f64, f64)> {
    let n = codes.len();
    let k = freqs.len();
    let mut sums = vec![0.0f64; k];
    (0..rows)
        .map(|r| {
            sums.iter_mut().for_each(|s| *s = 0.0);
            let row = &block[r * n..(r + 1) * n];
            for (j, &d) in row.iter().enumerate() {
                sums[codes[j] as usize] += d;
            }
            let own = codes[start + r] as usize;
            let intra = sums[own];
            let inter = (0..k)
                .filter(|&c| c != own)
                .map(|c| sums[c] / freqs[c] as f64)
                .fold(f64::INFINITY, f64::min);
            (intra, inter)
        })
        .collect()
}

/// Finish the silhouette from the per-row `(intra_sum, inter)` pairs: `intra`
/// is divided by `|own cluster| - 1`, then `s = (b - a) / max(a, b)`, with
/// sklearn's `nan_to_num` (a singleton cluster's `0/0` becomes 0).
pub fn silhouette_finish(pairs: &[(f64, f64)], codes: &[u32], freqs: &[u64]) -> Vec<f64> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, &(intra_sum, inter))| {
            let denom = freqs[codes[i] as usize].saturating_sub(1) as f64;
            let a = intra_sum / denom;
            let s = (inter - a) / a.max(inter);
            if s.is_nan() {
                0.0
            } else if s.is_infinite() {
                if s > 0.0 { f64::MAX } else { f64::MIN }
            } else {
                s
            }
        })
        .collect()
}

/// Label encoding + frequency table for the silhouette, with sklearn's label
/// count check.
pub fn silhouette_labels(labels: &[i64]) -> Result<(Vec<u32>, Vec<u64>), MetricError> {
    let (codes, k) = encode_checked(labels)?;
    let mut freqs = vec![0u64; k];
    for &c in &codes {
        freqs[c as usize] += 1;
    }
    Ok((codes, freqs))
}

/// `silhouette_samples` over a FULL precomputed row-major `n × n` distance
/// matrix (the `metric='precomputed'` path, and the reference for the blocked
/// one).
pub fn silhouette_samples_from_distances(distances: &[f64], labels: &[i64]) -> Result<Vec<f64>, MetricError> {
    let n = labels.len();
    if distances.len() != n * n {
        return Err(MetricError::BadShape);
    }
    let (codes, freqs) = silhouette_labels(labels)?;
    let pairs = silhouette_block(distances, n, 0, &codes, &freqs);
    Ok(silhouette_finish(&pairs, &codes, &freqs))
}

fn check_features(x: &[f64], n: usize) -> Result<usize, MetricError> {
    if n == 0 || x.len() % n != 0 {
        return Err(MetricError::BadShape);
    }
    Ok(x.len() / n)
}

/// Per-cluster centroids (row-major `k × d`) and sizes.
fn centroids(x: &[f64], d: usize, codes: &[u32], k: usize) -> (Vec<f64>, Vec<u64>) {
    let mut c = vec![0.0f64; k * d];
    let mut sizes = vec![0u64; k];
    for (i, &code) in codes.iter().enumerate() {
        let code = code as usize;
        sizes[code] += 1;
        for f in 0..d {
            c[code * d + f] += x[i * d + f];
        }
    }
    for code in 0..k {
        for f in 0..d {
            c[code * d + f] /= sizes[code] as f64;
        }
    }
    (c, sizes)
}

/// `calinski_harabasz_score` (variance-ratio criterion) over a row-major
/// `n × d` matrix.
pub fn calinski_harabasz_score(x: &[f64], labels: &[i64]) -> Result<f64, MetricError> {
    let n = labels.len();
    let d = check_features(x, n)?;
    let (codes, k) = encode_checked(labels)?;
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        for f in 0..d {
            mean[f] += x[i * d + f];
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    let (cent, sizes) = centroids(x, d, &codes, k);
    let mut extra = 0.0f64;
    for c in 0..k {
        let s: f64 = (0..d).map(|f| (cent[c * d + f] - mean[f]).powi(2)).sum();
        extra += sizes[c] as f64 * s;
    }
    let mut intra = 0.0f64;
    for (i, &code) in codes.iter().enumerate() {
        let c = code as usize;
        intra += (0..d).map(|f| (x[i * d + f] - cent[c * d + f]).powi(2)).sum::<f64>();
    }
    Ok(if intra == 0.0 {
        1.0
    } else {
        extra * (n - k) as f64 / (intra * (k as f64 - 1.0))
    })
}

/// `davies_bouldin_score` over a row-major `n × d` matrix (Euclidean).
pub fn davies_bouldin_score(x: &[f64], labels: &[i64]) -> Result<f64, MetricError> {
    let n = labels.len();
    let d = check_features(x, n)?;
    let (codes, k) = encode_checked(labels)?;
    let (cent, sizes) = centroids(x, d, &codes, k);
    let mut intra = vec![0.0f64; k];
    for (i, &code) in codes.iter().enumerate() {
        let c = code as usize;
        intra[c] += (0..d)
            .map(|f| (x[i * d + f] - cent[c * d + f]).powi(2))
            .sum::<f64>()
            .sqrt();
    }
    for c in 0..k {
        intra[c] /= sizes[c] as f64;
    }
    let mut cd = vec![0.0f64; k * k];
    for a in 0..k {
        for b in 0..k {
            cd[a * k + b] = (0..d)
                .map(|f| (cent[a * d + f] - cent[b * d + f]).powi(2))
                .sum::<f64>()
                .sqrt();
        }
    }
    let close = |v: f64| v.abs() <= 1e-8;
    if intra.iter().all(|&v| close(v)) || cd.iter().all(|&v| close(v)) {
        return Ok(0.0);
    }
    let mut total = 0.0f64;
    for a in 0..k {
        let mut best = f64::NEG_INFINITY;
        for b in 0..k {
            let dist = if cd[a * k + b] == 0.0 { f64::INFINITY } else { cd[a * k + b] };
            best = best.max((intra[a] + intra[b]) / dist);
        }
        total += best;
    }
    Ok(total / k as f64)
}

/// Minimum-cost assignment of a row-major `rows × cols` cost matrix (the
/// rectangular Hungarian algorithm, shortest augmenting paths with
/// potentials, O(r²c)). Returns `(row, col)` pairs, one per row of the smaller
/// side — scipy's `linear_sum_assignment` contract, whose optimum VALUE this
/// matches exactly (tied optima may pick different pairs).
pub fn linear_sum_assignment(cost: &[f64], rows: usize, cols: usize) -> Result<Vec<(usize, usize)>, MetricError> {
    if rows * cols != cost.len() {
        return Err(MetricError::BadShape);
    }
    if cost.iter().any(|v| !v.is_finite()) {
        return Err(MetricError::DomainError);
    }
    // Work on the orientation with rows <= cols.
    let transposed = rows > cols;
    let (r, c) = if transposed { (cols, rows) } else { (rows, cols) };
    let at = |i: usize, j: usize| if transposed { cost[j * cols + i] } else { cost[i * cols + j] };
    // 1-indexed potentials / matching, the classic e-maxx formulation.
    let mut u = vec![0.0f64; r + 1];
    let mut v = vec![0.0f64; c + 1];
    let mut p = vec![0usize; c + 1];
    let mut way = vec![0usize; c + 1];
    for i in 1..=r {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; c + 1];
        let mut used = vec![false; c + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=c {
                if !used[j] {
                    let cur = at(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=c {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut pairs: Vec<(usize, usize)> = (1..=c)
        .filter(|&j| p[j] != 0)
        .map(|j| {
            let (i, jj) = (p[j] - 1, j - 1);
            if transposed { (jj, i) } else { (i, jj) }
        })
        .collect();
    pairs.sort_unstable();
    Ok(pairs)
}

/// `consensus_score` from a precomputed row-major `n_a × n_b` similarity
/// matrix: the best one-to-one matching's total similarity over
/// `max(n_a, n_b)`.
pub fn consensus_from_similarity(sim: &[f64], n_a: usize, n_b: usize) -> Result<f64, MetricError> {
    if n_a == 0 || n_b == 0 {
        return Err(MetricError::EmptyInput);
    }
    let cost: Vec<f64> = sim.iter().map(|&s| 1.0 - s).collect();
    let pairs = linear_sum_assignment(&cost, n_a, n_b)?;
    let total: f64 = pairs.iter().map(|&(i, j)| sim[i * n_b + j]).sum();
    Ok(total / n_a.max(n_b) as f64)
}

/// `consensus_score(a, b, similarity='jaccard')`. Each bicluster set is given
/// as row-major boolean (0/1) membership matrices: `a_rows` is `n_a × R`,
/// `a_cols` is `n_a × C`, and likewise for `b`.
#[allow(clippy::too_many_arguments)]
pub fn consensus_score_jaccard(
    a_rows: &[f64],
    a_cols: &[f64],
    b_rows: &[f64],
    b_cols: &[f64],
    n_a: usize,
    n_b: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<f64, MetricError> {
    if a_rows.len() != n_a * n_rows
        || b_rows.len() != n_b * n_rows
        || a_cols.len() != n_a * n_cols
        || b_cols.len() != n_b * n_cols
    {
        return Err(MetricError::BadShape);
    }
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    let sum = |x: &[f64]| x.iter().sum::<f64>();
    let mut sim = vec![0.0f64; n_a * n_b];
    for i in 0..n_a {
        let ar = &a_rows[i * n_rows..(i + 1) * n_rows];
        let ac = &a_cols[i * n_cols..(i + 1) * n_cols];
        for j in 0..n_b {
            let br = &b_rows[j * n_rows..(j + 1) * n_rows];
            let bc = &b_cols[j * n_cols..(j + 1) * n_cols];
            let inter = dot(ar, br) * dot(ac, bc);
            let a_size = sum(ar) * sum(ac);
            let b_size = sum(br) * sum(bc);
            sim[i * n_b + j] = inter / (a_size + b_size - inter);
        }
    }
    consensus_from_similarity(&sim, n_a, n_b)
}

/// `silhouette_samples` over a row-major `n × d` feature matrix with any
/// [`super::pairwise::Distance`]. The distances are computed one row block at
/// a time (sized to keep each block near 64 MiB, sklearn's
/// `pairwise_distances_chunked` idea) and reduced immediately, so peak memory
/// is one block, not the `n × n` matrix.
pub fn silhouette_samples(
    x: &[f64],
    d: usize,
    labels: &[i64],
    metric: &super::pairwise::Distance,
) -> Result<Vec<f64>, MetricError> {
    let n = labels.len();
    if d == 0 || x.len() != n * d {
        return Err(MetricError::BadShape);
    }
    let (codes, freqs) = silhouette_labels(labels)?;
    let block = ((64usize << 20) / 8 / n.max(1)).clamp(1, n.max(1));
    let mut pairs = Vec::with_capacity(n);
    for start in (0..n).step_by(block) {
        let stop = (start + block).min(n);
        let mut dist = super::pairwise::pairwise_distances(&x[start * d..stop * d], Some(x), d, metric)?;
        // `Y = X` semantics for the diagonal, which the blocked call (with an
        // explicit `Y`) cannot know about.
        for r in 0..stop - start {
            dist[r * n + start + r] = 0.0;
        }
        pairs.extend(silhouette_block(&dist, stop - start, start, &codes, &freqs));
    }
    Ok(silhouette_finish(&pairs, &codes, &freqs))
}
