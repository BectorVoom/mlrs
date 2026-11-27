//! Variational document inference for latent Dirichlet allocation.
//!
//! The document-word matrix is stored in CSR form. Topic probabilities and
//! document distributions are dense, with topics contiguous within each word
//! or document respectively.

/// Document inference inputs. Matrix dimensions are validated by the caller.
pub struct DocumentBatch<'a> {
    pub offsets: &'a [usize],
    pub indices: &'a [usize],
    pub counts: &'a [f64],
    pub word_topics: &'a [f64],
    pub topics: usize,
}

/// Variational E-step for a batch. Update `distributions` in place and
/// optionally return expected topic-word counts in word-major layout.
pub fn infer(
    batch: DocumentBatch<'_>,
    distributions: &mut [f64],
    prior: f64,
    max_iter: usize,
    tol: f64,
    epsilon: f64,
    collect_stats: bool,
) -> Vec<f64> {
    let DocumentBatch {
        offsets,
        indices,
        counts,
        word_topics,
        topics,
    } = batch;
    let mut stats = if collect_stats {
        vec![0.0; word_topics.len()]
    } else {
        Vec::new()
    };
    let mut expected = vec![0.0; topics];
    let mut next = vec![0.0; topics];
    for (span, row) in offsets
        .windows(2)
        .zip(distributions.chunks_exact_mut(topics))
    {
        let range = span[0]..span[1];
        let words = &indices[range.clone()];
        let counts = &counts[range];
        let sum = digamma(row.iter().sum());
        for (e, &value) in expected.iter_mut().zip(row.iter()) {
            *e = (digamma(value) - sum).exp();
        }
        for _ in 0..max_iter {
            next.fill(0.0);
            for (&word, &count) in words.iter().zip(counts) {
                let weights = &word_topics[word * topics..(word + 1) * topics];
                let norm = dot(&expected, weights) + epsilon;
                for (target, &weight) in next.iter_mut().zip(weights) {
                    *target += weight * (count / norm);
                }
            }
            let mut change = 0.0;
            for ((value, &accumulated), &e) in row.iter_mut().zip(&next).zip(&expected) {
                let updated = e * accumulated + prior;
                change += (updated - *value).abs();
                *value = updated;
            }
            let sum = digamma(row.iter().sum());
            for (e, &value) in expected.iter_mut().zip(row.iter()) {
                *e = (digamma(value) - sum).exp();
            }
            if change / (topics as f64) < tol {
                break;
            }
        }
        if collect_stats {
            for (&word, &count) in words.iter().zip(counts) {
                let weights = &word_topics[word * topics..(word + 1) * topics];
                let norm = dot(&expected, weights) + epsilon;
                for ((target, &e), &weight) in stats[word * topics..(word + 1) * topics]
                    .iter_mut()
                    .zip(&expected)
                    .zip(weights)
                {
                    *target += e * (count / norm) * weight;
                }
            }
        }
    }
    stats
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut offset = 0.0;
    while x < 8.0 {
        offset -= x.recip();
        x += 1.0;
    }
    let r = x.recip();
    let z = r * r;
    offset + x.ln() - r / 2.0 - z * (1.0 / 12.0 - z * (1.0 / 120.0 - z * (1.0 / 252.0 - z / 240.0)))
}
