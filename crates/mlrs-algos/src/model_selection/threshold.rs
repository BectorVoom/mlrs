//! Decision-threshold tuning (MODSEL-RS-07).
//!
//! `FixedThresholdClassifier` and `TunedThresholdClassifierCV` both answer the
//! same question — "at what score does a positive prediction start?" — and
//! neither needs to touch the estimator to do the arithmetic. The classifier
//! produces per-row scores; everything after that is here.
//!
//! ## Why interpolation, not a shared threshold grid
//!
//! Each CV fold produces its OWN threshold vector (the distinct scores its
//! validation rows happened to take), so the folds cannot be averaged
//! elementwise. sklearn builds one common grid spanning every fold's range,
//! linearly interpolates each fold's score curve onto it, and averages there.
//! Skipping the interpolation and averaging fold curves by position would
//! silently compare thresholds that are not the same number.

use super::{value_err, Result};

/// Apply a decision threshold to per-row scores
/// (`FixedThresholdClassifier.predict`).
///
/// The comparison is `>= threshold`, matching sklearn's
/// `(y_score >= threshold).astype(int)` — a row scoring *exactly* the
/// threshold is POSITIVE. Returns indices into `classes`: `0` for the negative
/// class, `1` for the positive one.
pub fn apply_threshold(scores: &[f64], threshold: f64) -> Vec<i64> {
    scores.iter().map(|&s| i64::from(s >= threshold)).collect()
}

/// `np.interp(x, xp, fp)` — piecewise-linear interpolation with numpy's
/// **clamped** ends.
///
/// Outside `xp`'s range numpy returns the nearest endpoint value rather than
/// extrapolating, which is load-bearing here: a fold whose thresholds cover a
/// narrower range than the common grid must contribute its edge score to the
/// mean, not a linear continuation of its last segment.
///
/// `xp` must be ascending, as `np.interp` documents (and as sklearn's
/// threshold vectors are by construction).
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>> {
    if xp.len() != fp.len() {
        return Err(value_err!(
            "interp: xp and fp have different lengths ({} vs {})",
            xp.len(),
            fp.len()
        ));
    }
    if xp.is_empty() {
        return Err(value_err!("interp: xp is empty"));
    }
    Ok(x.iter()
        .map(|&q| {
            if q <= xp[0] {
                return fp[0];
            }
            if q >= xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }
            // First index whose xp exceeds q; the segment is [i-1, i].
            let i = xp.partition_point(|&v| v <= q);
            let (x0, x1) = (xp[i - 1], xp[i]);
            let (y0, y1) = (fp[i - 1], fp[i]);
            if x1 == x0 {
                y1
            } else {
                y0 + (y1 - y0) * (q - x0) / (x1 - x0)
            }
        })
        .collect())
}

/// `np.linspace(start, stop, num)` — `num` points INCLUSIVE of both ends.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    (0..num)
        .map(|i| {
            if i == num - 1 {
                // Compute the last point directly: accumulating `num - 1`
                // steps drifts off `stop` by an ulp or two, and the tuned
                // threshold is reported to the user verbatim.
                stop
            } else {
                start + step * i as f64
            }
        })
        .collect()
}

/// One fold's threshold/score curve.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldCurve {
    /// Ascending thresholds this fold produced.
    pub thresholds: Vec<f64>,
    /// The objective score at each threshold.
    pub scores: Vec<f64>,
}

/// The tuned outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct TunedThreshold {
    pub best_threshold: f64,
    pub best_score: f64,
    /// The common threshold grid (`cv_results_["thresholds"]`).
    pub thresholds: Vec<f64>,
    /// The interpolated mean objective at each grid point
    /// (`cv_results_["scores"]`).
    pub scores: Vec<f64>,
}

/// How the common threshold grid is chosen — `TunedThresholdClassifierCV`'s
/// `thresholds` parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdGrid {
    /// An int: `linspace` over the union of every fold's threshold range.
    Count(usize),
    /// An explicit array of thresholds.
    Explicit(Vec<f64>),
}

/// `TunedThresholdClassifierCV.fit`'s reduction: build the common grid,
/// interpolate every fold onto it, average, and take the argmax.
///
/// Rejects a constant-prediction estimator with sklearn's message — a fold
/// whose first and last thresholds coincide carries no information to optimize
/// over, and interpolating it would quietly contribute a flat line to the mean.
pub fn tune_threshold(folds: &[FoldCurve], grid: &ThresholdGrid) -> Result<TunedThreshold> {
    if folds.is_empty() {
        return Err(value_err!("no folds were scored"));
    }
    for fold in folds {
        if fold.thresholds.is_empty() {
            return Err(value_err!("a fold produced no thresholds"));
        }
        let (first, last) = (
            fold.thresholds[0],
            fold.thresholds[fold.thresholds.len() - 1],
        );
        if (first - last).abs() <= 1e-8 + 1e-5 * last.abs() {
            return Err(value_err!(
                "The provided estimator makes constant predictions. Therefore, \
                 it is impossible to optimize the decision threshold."
            ));
        }
    }

    let thresholds = match grid {
        ThresholdGrid::Explicit(values) => values.clone(),
        ThresholdGrid::Count(num) => {
            let min_threshold = folds
                .iter()
                .map(|f| f.thresholds.iter().copied().fold(f64::INFINITY, f64::min))
                .fold(f64::INFINITY, f64::min);
            let max_threshold = folds
                .iter()
                .map(|f| {
                    f.thresholds
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            linspace(min_threshold, max_threshold, *num)
        }
    };
    if thresholds.is_empty() {
        return Err(value_err!("the threshold grid is empty"));
    }

    let mut mean = vec![0.0f64; thresholds.len()];
    for fold in folds {
        let interpolated = interp(&thresholds, &fold.thresholds, &fold.scores)?;
        for (m, v) in mean.iter_mut().zip(&interpolated) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= folds.len() as f64;
    }

    // `argmax`: the FIRST maximum wins, matching numpy.
    let mut best_idx = 0usize;
    for (i, &v) in mean.iter().enumerate() {
        if v > mean[best_idx] {
            best_idx = i;
        }
    }
    Ok(TunedThreshold {
        best_threshold: thresholds[best_idx],
        best_score: mean[best_idx],
        thresholds,
        scores: mean,
    })
}
