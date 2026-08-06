//! Cross-validation aggregation, curve schedules and the permutation test
//! (MODSEL-RS-05).
//!
//! `cross_validate`, `learning_curve`, `validation_curve` and
//! `permutation_test_score` are all the same shape: schedule some (train,
//! test) work, let the estimator produce a number for each unit, then reduce.
//! The scheduling and the reduction live here; the estimator call is the
//! caller's.
//!
//! Every reduction reproduces sklearn's exact arithmetic, including the two
//! places it is easy to get subtly wrong:
//!
//! * the score **std** is a population std (`ddof=0`), so it is smaller than
//!   the sample std a `Vec::iter().std()` helper would hand back;
//! * **NaN scores** (a fold whose fit failed under `error_score=np.nan`) are
//!   ranked as *worse than the worst finite score* rather than dropped or
//!   propagated, so a search still produces a usable ranking.

use super::{value_err, Result};

/// The per-candidate reduction of a `(n_candidates, n_splits)` score matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreSummary {
    /// `mean_test_score` — the mean over splits, per candidate.
    pub mean: Vec<f64>,
    /// `std_test_score` — the POPULATION std over splits, per candidate.
    pub std: Vec<f64>,
    /// `rank_test_score` — 1 is best; ties share the lower rank
    /// (`rankdata(method="min")`).
    pub rank: Vec<i32>,
}

/// Reduce a row-major `(n_candidates, n_splits)` score matrix the way
/// `BaseSearchCV._format_results` does.
pub fn summarize_scores(
    scores: &[f64],
    n_candidates: usize,
    n_splits: usize,
) -> Result<ScoreSummary> {
    if scores.len() != n_candidates * n_splits {
        return Err(value_err!(
            "score matrix has {} entries, expected n_candidates * n_splits = {}",
            scores.len(),
            n_candidates * n_splits
        ));
    }
    let mut mean = Vec::with_capacity(n_candidates);
    let mut std = Vec::with_capacity(n_candidates);
    for c in 0..n_candidates {
        let row = &scores[c * n_splits..(c + 1) * n_splits];
        // `np.mean` over a row containing NaN is NaN — sklearn relies on that
        // to mark a candidate whose folds did not all score.
        let m = row.iter().sum::<f64>() / n_splits as f64;
        let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n_splits as f64;
        mean.push(m);
        std.push(var.sqrt());
    }
    let rank = rank_scores(&mean);
    Ok(ScoreSummary { mean, std, rank })
}

/// `rankdata(-scores, method="min")` with sklearn's NaN handling: a NaN is
/// replaced by `nanmin(scores) - 1` first, so it ranks last but still ranks.
///
/// `method="min"` means tied candidates all take the SMALLEST rank they could
/// (two candidates tied for best are both rank 1, and the next is rank 3), so
/// ranks are not a permutation of `1..=n`.
pub fn rank_scores(scores: &[f64]) -> Vec<i32> {
    let finite_min = scores
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::INFINITY, f64::min);
    let filled: Vec<f64> = scores
        .iter()
        .map(|&v| {
            if v.is_nan() {
                if finite_min.is_finite() {
                    finite_min - 1.0
                } else {
                    // Every score is NaN: they are all equally (un)ranked.
                    0.0
                }
            } else {
                v
            }
        })
        .collect();

    // Rank on the NEGATED score so the highest score is rank 1.
    let mut order: Vec<usize> = (0..filled.len()).collect();
    order.sort_by(|&a, &b| {
        (-filled[a])
            .partial_cmp(&-filled[b])
            .expect("NaNs were replaced above")
            .then(a.cmp(&b))
    });

    let mut rank = vec![0i32; filled.len()];
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && filled[order[j + 1]] == filled[order[i]] {
            j += 1;
        }
        // `method="min"`: everything in this tie block takes the block's first
        // (1-based) position.
        for &idx in &order[i..=j] {
            rank[idx] = i as i32 + 1;
        }
        i = j + 1;
    }
    rank
}

/// `learning_curve`'s `train_sizes` parameter: either fractions of the maximum
/// training-set size or absolute row counts.
///
/// The two are NOT interchangeable — sklearn dispatches on the array's dtype,
/// so `[1.0]` means "all of it" while `[1]` means "one row".
#[derive(Debug, Clone, PartialEq)]
pub enum TrainSizes {
    Fractions(Vec<f64>),
    Absolute(Vec<usize>),
}

/// sklearn's `_translate_train_sizes` — resolve `train_sizes` to absolute,
/// deduplicated, ascending row counts.
///
/// Returns the sizes plus sklearn's `RuntimeWarning` text if deduplication
/// removed a tick (two fractions can round to the same row count, which
/// silently shortens the curve).
pub fn translate_train_sizes(
    train_sizes: &TrainSizes,
    n_max_training_samples: usize,
) -> Result<(Vec<usize>, Option<String>)> {
    let n_ticks;
    let mut absolute: Vec<i64> = match train_sizes {
        TrainSizes::Fractions(fracs) => {
            n_ticks = fracs.len();
            let lo = fracs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = fracs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if lo <= 0.0 || hi > 1.0 {
                return Err(value_err!(
                    "train_sizes has been interpreted as fractions of the \
                     maximum number of training samples and must be within \
                     (0, 1], but is within [{lo:.6}, {hi:.6}]."
                ));
            }
            fracs
                .iter()
                // `.astype(int)` TRUNCATES toward zero, then sklearn clips into
                // [1, n_max] — so a tiny fraction becomes 1 rather than 0.
                .map(|f| {
                    let v = (f * n_max_training_samples as f64) as i64;
                    v.clamp(1, n_max_training_samples as i64)
                })
                .collect()
        }
        TrainSizes::Absolute(sizes) => {
            n_ticks = sizes.len();
            let lo = sizes.iter().copied().min().unwrap_or(0);
            let hi = sizes.iter().copied().max().unwrap_or(0);
            if lo == 0 || hi > n_max_training_samples {
                return Err(value_err!(
                    "train_sizes has been interpreted as absolute numbers of \
                     training samples and must be within (0, \
                     {n_max_training_samples}], but is within [{lo}, {hi}]."
                ));
            }
            sizes.iter().map(|&s| s as i64).collect()
        }
    };

    absolute.sort_unstable();
    absolute.dedup();
    let warning = (n_ticks > absolute.len()).then(|| {
        format!(
            "Removed duplicate entries from 'train_sizes'. Number of ticks \
             will be less than the size of 'train_sizes': {} instead of \
             {n_ticks}.",
            absolute.len()
        )
    });
    Ok((absolute.into_iter().map(|v| v as usize).collect(), warning))
}

/// `permutation_test_score`'s p-value: `(C + 1) / (n_permutations + 1)` where
/// `C` counts permutation scores at least as good as the true score.
///
/// The `+1`s are not a smoothing choice to second-guess — they make the
/// best attainable p-value `1 / (n_permutations + 1)` rather than 0, which is
/// what keeps the statistic honest about how many permutations were actually
/// run.
pub fn permutation_pvalue(score: f64, permutation_scores: &[f64]) -> f64 {
    let c = permutation_scores.iter().filter(|&&s| s >= score).count();
    (c as f64 + 1.0) / (permutation_scores.len() as f64 + 1.0)
}

/// Verify that a set of test index vectors partitions `0..n_samples` — the
/// precondition `cross_val_predict` needs to assemble one prediction per row.
///
/// sklearn raises `ValueError("cross_val_predict only works for partitions")`
/// for a splitter like `ShuffleSplit` that tests some rows twice and others
/// never; this reports the same condition.
pub fn check_is_partition(test_sets: &[Vec<i64>], n_samples: usize) -> Result<()> {
    let mut seen = vec![false; n_samples];
    for test in test_sets {
        for &idx in test {
            if idx < 0 || idx as usize >= n_samples {
                return Err(value_err!(
                    "test index {idx} is out of range for {n_samples} samples"
                ));
            }
            if seen[idx as usize] {
                return Err(value_err!("cross_val_predict only works for partitions"));
            }
            seen[idx as usize] = true;
        }
    }
    if !seen.iter().all(|&s| s) {
        return Err(value_err!("cross_val_predict only works for partitions"));
    }
    Ok(())
}

/// The inverse permutation that reassembles `cross_val_predict`'s output.
///
/// Predictions come back in fold order (`test_sets[0]`'s rows, then
/// `test_sets[1]`'s, …); this returns, for each original row, its position in
/// that concatenated buffer, so the caller can scatter in one pass instead of
/// searching.
pub fn partition_inverse(test_sets: &[Vec<i64>], n_samples: usize) -> Result<Vec<usize>> {
    check_is_partition(test_sets, n_samples)?;
    let mut inverse = vec![0usize; n_samples];
    let mut position = 0usize;
    for test in test_sets {
        for &idx in test {
            inverse[idx as usize] = position;
            position += 1;
        }
    }
    Ok(inverse)
}
