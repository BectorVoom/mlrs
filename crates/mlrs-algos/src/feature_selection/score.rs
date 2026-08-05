//! The `sklearn.feature_selection` SCORING FUNCTIONS (FSEL-01).
//!
//! sklearn exposes seven module-level scoring functions. Five have closed forms
//! and live here; the two `mutual_info_*` estimators are k-nearest-neighbour
//! entropy estimators and live in [`super::mutual_info`].
//!
//! | function        | returns              | mlrs entry point           |
//! |-----------------|----------------------|----------------------------|
//! | `f_oneway`      | `(F, p)` per column  | [`f_oneway`]               |
//! | `f_classif`     | `(F, p)` per column  | [`f_classif`]              |
//! | `chi2`          | `(χ², p)` per column | [`chi2`]                   |
//! | `r_regression`  | `r` per column       | [`r_regression`]           |
//! | `f_regression`  | `(F, p)` per column  | [`f_regression`]           |
//!
//! Every one is assembled from ONE of
//! [`prims::feature_score`](mlrs_backend::prims::feature_score)'s `f64` column
//! sweeps plus [`prims::special`](mlrs_backend::prims::special)'s distribution
//! tails, so the whole cost is one `O(n·d)` pass and `O(d)` scalar work — see
//! those modules' docs for why the accumulation is host `f64` on every backend.
//!
//! ## Degenerate columns are the specification, not an edge case
//! Feature selection is applied to real, messy design matrices, and a constant
//! column, an all-zero column, or a perfectly correlated column is the normal
//! case rather than the exception. sklearn has a precise (and slightly
//! surprising) answer for each, and it is load-bearing because the selectors'
//! masks depend on it:
//!
//! * `f_oneway`/`f_classif` on a column that is constant WITHIN every class
//!   divides `msb/msw = 0/0` and produces `NaN` — which `_clean_nans` later maps
//!   to `f64::MIN` so the column sorts LAST. mlrs reproduces the `NaN`, and
//!   [`super::univariate`] reproduces the `_clean_nans` mapping.
//! * `chi2` on an all-zero column divides `0/0` per class and also yields `NaN`.
//! * `f_regression`'s `force_finite=true` (the default) rewrites `±inf` F to
//!   `f64::MAX` with p-value `0`, and `NaN` F to `0.0` with p-value `1.0`.
//!   `force_finite=false` leaves both alone. Both are reproduced exactly,
//!   including the `f64::MAX` sentinel, because a caller comparing against
//!   sklearn's `scores_` sees that specific number.
//! * `r_regression`'s `force_finite=true` maps a `NaN` correlation to `0.0`.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::sync::Arc;

use mlrs_backend::prims::feature_score::{class_col_sums, cross_moments, ClassColSums};
use mlrs_backend::prims::special::{chi2_sf, f_sf};

use crate::error::AlgoError;

/// What a scoring function returns: one score per feature, and optionally one
/// p-value per feature.
///
/// The `Option` is the whole reason sklearn's `_BaseFilter.fit` branches on
/// `isinstance(score_func_ret, (list, tuple))`: `r_regression` and both
/// `mutual_info_*` functions return scores ONLY, and the three p-value-based
/// selectors cannot be used with them. [`super::univariate`] enforces that with
/// [`AlgoError::ScoreFuncHasNoPValues`] rather than panicking.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreResult {
    /// One score per feature, in column order.
    pub scores: Vec<f64>,
    /// One p-value per feature, or `None` for a scores-only function.
    pub pvalues: Option<Vec<f64>>,
}

impl ScoreResult {
    /// A scores-only result (`r_regression`, `mutual_info_*`).
    pub fn scores_only(scores: Vec<f64>) -> Self {
        Self {
            scores,
            pvalues: None,
        }
    }

    /// A `(scores, pvalues)` result (`f_classif`, `chi2`, `f_regression`).
    pub fn with_pvalues(scores: Vec<f64>, pvalues: Vec<f64>) -> Self {
        Self {
            scores,
            pvalues: Some(pvalues),
        }
    }
}

/// A user-supplied scoring function, the Rust analogue of sklearn's
/// `score_func` callable.
///
/// The signature takes the design as a row-major `f64` host slice with its
/// `(n_samples, n_features)` geometry and the target as a length-`n_samples`
/// slice — deliberately CONCRETE `f64` rather than generic over the estimator's
/// `F`, for the same reason the built-in scores accumulate in `f64` (see the
/// prim's docs): a score function's output feeds an exponentially sensitive
/// p-value, so widening once at the boundary is strictly better than letting
/// every custom implementation decide.
///
/// `Arc<dyn Fn>` rather than a bare `&dyn Fn` so a selector can OWN its score
/// function and stay `Clone` + `'static`, which the typestate `Fitted` value and
/// the PyO3 wrapper both need.
pub type CustomScoreFunc =
    Arc<dyn Fn(&[f64], &[f64], usize, usize) -> Result<ScoreResult, AlgoError> + Send + Sync>;

/// Which scoring function a univariate selector uses — sklearn's `score_func`
/// parameter.
///
/// The built-ins are enum variants rather than function pointers so a selector
/// is `Clone`/`Debug` and so the PyO3 layer can select one by name without
/// crossing a function-pointer boundary; [`ScoreFunc::Custom`] covers the
/// arbitrary-callable case sklearn's signature admits.
#[derive(Clone)]
pub enum ScoreFunc {
    /// `sklearn.feature_selection.f_classif` — the sklearn DEFAULT for every
    /// univariate selector.
    FClassif,
    /// `sklearn.feature_selection.chi2`. Requires non-negative `X`.
    Chi2,
    /// `sklearn.feature_selection.r_regression`. Scores only, no p-values.
    RRegression {
        /// Center `X` and `y` before correlating (sklearn default `true`).
        center: bool,
        /// Map a `NaN` correlation (constant column or constant target) to
        /// `0.0` (sklearn default `true`).
        force_finite: bool,
    },
    /// `sklearn.feature_selection.f_regression`.
    FRegression {
        /// Center `X` and `y` before correlating (sklearn default `true`).
        center: bool,
        /// Rewrite `±inf` F to `f64::MAX`/p `0` and `NaN` F to `0`/p `1`
        /// (sklearn default `true`).
        force_finite: bool,
    },
    /// `sklearn.feature_selection.mutual_info_classif`. Scores only.
    MutualInfoClassif(super::mutual_info::MutualInfoParams),
    /// `sklearn.feature_selection.mutual_info_regression`. Scores only.
    MutualInfoRegression(super::mutual_info::MutualInfoParams),
    /// An arbitrary caller-supplied score function.
    Custom(CustomScoreFunc),
}

impl std::fmt::Debug for ScoreFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FClassif => f.write_str("f_classif"),
            Self::Chi2 => f.write_str("chi2"),
            Self::RRegression {
                center,
                force_finite,
            } => write!(
                f,
                "r_regression(center={center}, force_finite={force_finite})"
            ),
            Self::FRegression {
                center,
                force_finite,
            } => write!(
                f,
                "f_regression(center={center}, force_finite={force_finite})"
            ),
            Self::MutualInfoClassif(p) => write!(f, "mutual_info_classif({p:?})"),
            Self::MutualInfoRegression(p) => write!(f, "mutual_info_regression({p:?})"),
            Self::Custom(_) => f.write_str("<custom score_func>"),
        }
    }
}

impl Default for ScoreFunc {
    /// sklearn's default for every univariate selector is `f_classif`.
    fn default() -> Self {
        Self::FClassif
    }
}

impl ScoreFunc {
    /// Whether this function produces p-values — i.e. whether it can be used
    /// with `SelectFpr` / `SelectFdr` / `SelectFwe`.
    ///
    /// [`ScoreFunc::Custom`] reports `true` optimistically: its output is only
    /// known once called, so the p-value requirement is checked against the
    /// actual [`ScoreResult`] instead (see [`super::univariate`]).
    pub fn yields_pvalues(&self) -> bool {
        !matches!(
            self,
            Self::RRegression { .. } | Self::MutualInfoClassif(_) | Self::MutualInfoRegression(_)
        )
    }

    /// Evaluate the score function on a row-major `f64` design and target.
    pub fn eval(&self, x: &[f64], y: &[f64], n: usize, d: usize) -> Result<ScoreResult, AlgoError> {
        match self {
            Self::FClassif => f_classif(x, y, n, d),
            Self::Chi2 => chi2(x, y, n, d),
            Self::RRegression {
                center,
                force_finite,
            } => Ok(ScoreResult::scores_only(r_regression(
                x,
                y,
                n,
                d,
                *center,
                *force_finite,
            )?)),
            Self::FRegression {
                center,
                force_finite,
            } => f_regression(x, y, n, d, *center, *force_finite),
            Self::MutualInfoClassif(p) => Ok(ScoreResult::scores_only(
                super::mutual_info::mutual_info_classif(x, y, n, d, p)?,
            )),
            Self::MutualInfoRegression(p) => Ok(ScoreResult::scores_only(
                super::mutual_info::mutual_info_regression(x, y, n, d, p)?,
            )),
            Self::Custom(f) => f(x, y, n, d),
        }
    }
}

// ===========================================================================
// f_oneway / f_classif
// ===========================================================================

/// Map a target vector to CLASS INDICES in `numpy.unique` order, returning the
/// indices and the sorted-unique class values.
///
/// `numpy.unique` order is load-bearing, not incidental: `f_classif` builds its
/// per-class groups as `[X[y == k] for k in np.unique(y)]`, and `chi2` builds
/// its `Y` indicator with `LabelBinarizer`, which also sorts. So the per-class
/// rows of the moment table line up with sklearn's group order only if the
/// mapping is sorted-unique — and `f_oneway`'s `ssbn` accumulates in that order,
/// which is what makes the last bits reproducible against it.
///
/// Comparison is by TOTAL float order (`f64::total_cmp`) so `NaN` targets sort
/// to one end deterministically rather than producing an inconsistent
/// comparator; a `NaN` target is rejected upstream by the shim's `check_array`
/// and by the Rust bridge, so this is a defensive ordering choice, not a
/// supported input.
pub(crate) fn class_indices(y: &[f64]) -> (Vec<u32>, Vec<f64>) {
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.total_cmp(b));
    classes.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let idx = y
        .iter()
        .map(|v| {
            classes
                .iter()
                .position(|c| c.to_bits() == v.to_bits())
                .expect("every target value is in the deduped class list") as u32
        })
        .collect();
    (idx, classes)
}

/// `sklearn.feature_selection.f_oneway` over pre-grouped column moments — the
/// one-way ANOVA F-statistic and its p-value, per column.
///
/// This is sklearn's `f_oneway` transcribed from raw moments rather than from
/// the group matrices, which is possible because every quantity it forms is a
/// sum or a sum of squares:
///
/// ```text
/// sstot = ss_alldata − (Σ_all)² / n
/// ssbn  = Σ_k (Σ_k x)² / n_k − (Σ_all)² / n
/// sswn  = sstot − ssbn
/// F     = (ssbn / (K − 1)) / (sswn / (n − K))
/// p     = fdtrc(K − 1, n − K, F)
/// ```
///
/// The subtraction order matters and is sklearn's verbatim: `sstot` and `ssbn`
/// each subtract `square_of_sums_alldata / n` SEPARATELY and `sswn` is their
/// difference, rather than the algebraically-equal single expression. On a
/// column with a large mean the two forms differ in their last bits, and the
/// F-statistic of a nearly-constant column — precisely the column whose score
/// decides a `SelectKBest` tie — is the ratio of two such cancelling
/// quantities. Matching sklearn's grouping is what keeps that ratio inside the
/// 1e-5 band instead of merely close.
pub fn f_oneway_from_moments(m: &ClassColSums) -> ScoreResult {
    let k = m.counts.len();
    let d = m.total_sum.len();
    let n: usize = m.counts.iter().sum();
    let n_f = n as f64;
    let dfbn = (k - 1) as f64;
    let dfwn = (n - k) as f64;

    let mut scores = Vec::with_capacity(d);
    let mut pvalues = Vec::with_capacity(d);
    for c in 0..d {
        let sq_sums_all = m.total_sum[c] * m.total_sum[c];
        let sstot = m.total_sumsq[c] - sq_sums_all / n_f;
        let mut ssbn = 0.0;
        for kk in 0..k {
            let s = m.sums[kk * d + c];
            // A class with no rows contributes nothing; sklearn cannot reach
            // this (its groups come from `np.unique(y)`, so every group is
            // non-empty) but a caller passing an explicit class list can.
            if m.counts[kk] > 0 {
                ssbn += s * s / m.counts[kk] as f64;
            }
        }
        ssbn -= sq_sums_all / n_f;
        let sswn = sstot - ssbn;
        let msb = ssbn / dfbn;
        let msw = sswn / dfwn;
        // `msw == 0` is the constant-within-every-class column: sklearn warns
        // ("Features %s are constant") and lets the division produce `NaN`
        // (0/0) or `inf` (positive/0). Both are reproduced — `_clean_nans` in
        // the selectors is what gives them their final ranking.
        let f = msb / msw;
        scores.push(f);
        pvalues.push(f_sf(f, dfbn, dfwn));
    }
    ScoreResult::with_pvalues(scores, pvalues)
}

/// `sklearn.feature_selection.f_classif(X, y)` — the ANOVA F-value of each
/// column against the class label.
pub fn f_classif(x: &[f64], y: &[f64], n: usize, d: usize) -> Result<ScoreResult, AlgoError> {
    let (labels, classes) = class_indices(y);
    let moments = class_col_sums(x, &labels, n, d, classes.len())?;
    Ok(f_oneway_from_moments(&moments))
}

/// `sklearn.feature_selection.f_oneway(*groups)` — the free-standing ANOVA over
/// explicitly-grouped samples.
///
/// sklearn's signature takes the groups as separate `(n_k, d)` matrices; the
/// Rust form takes them as a slice of `(rows, row_major_values)` pairs, which is
/// the same information without a variadic. Every group must have the same
/// column count `d`.
///
/// This exists because `f_oneway` is PUBLIC in sklearn's `__init__.py` — it is
/// not merely `f_classif`'s helper, and a caller porting code that calls it
/// directly needs it.
pub fn f_oneway(groups: &[(usize, &[f64])], d: usize) -> Result<ScoreResult, AlgoError> {
    if groups.len() < 2 || d == 0 {
        return Err(AlgoError::InvalidSelectorParam {
            estimator: "f_oneway",
            param: "groups",
            value: groups.len() as f64,
            reason: "the one-way ANOVA needs at least 2 groups of at least 1 column",
        });
    }
    let mut m = ClassColSums {
        counts: vec![0; groups.len()],
        sums: vec![0.0; groups.len() * d],
        total_sum: vec![0.0; d],
        total_sumsq: vec![0.0; d],
    };
    for (g, &(rows, values)) in groups.iter().enumerate() {
        if values.len() != rows * d {
            return Err(AlgoError::Prim(mlrs_core::PrimError::ShapeMismatch {
                operand: "group",
                rows,
                cols: d,
                len: values.len(),
            }));
        }
        m.counts[g] = rows;
        for r in 0..rows {
            for c in 0..d {
                let v = values[r * d + c];
                m.sums[g * d + c] += v;
                m.total_sum[c] += v;
                m.total_sumsq[c] += v * v;
            }
        }
    }
    if m.counts.iter().sum::<usize>() <= groups.len() {
        return Err(AlgoError::InvalidSelectorParam {
            estimator: "f_oneway",
            param: "groups",
            value: m.counts.iter().sum::<usize>() as f64,
            reason: "total sample count must exceed the group count (dfwn > 0)",
        });
    }
    Ok(f_oneway_from_moments(&m))
}

// ===========================================================================
// chi2
// ===========================================================================

/// `sklearn.feature_selection.chi2(X, y)` — the χ² statistic of each
/// non-negative column against the class label, and its p-value.
///
/// sklearn forms the `n_classes × n_features` `observed` table as `Yᵀ X` (with
/// `Y` the label indicator) and the `expected` table as
/// `class_prob.T @ feature_count`, then reduces
/// `Σ_k (observed − expected)² / expected` down the class axis with
/// `df = n_classes − 1`. Every input to that is a per-class column sum, so the
/// whole score follows from one [`class_col_sums`] sweep.
///
/// ## The BINARY-target special case is not optional
/// `LabelBinarizer` on a 2-class target returns ONE column, and sklearn then
/// does `Y = np.append(1 - Y, Y, axis=1)` to restore two. The `df` it passes to
/// `chdtrc` is `k − 1` where `k = len(f_obs)` — the number of ROWS of the
/// observed table, i.e. `2` for a binary target, giving `df = 1`. Deriving `df`
/// from the class count directly gives the same answer here only because the
/// restored indicator has exactly `n_classes` rows; the equality is worth
/// stating because for a ONE-class target sklearn's `LabelBinarizer` also
/// returns one column and the append makes it two, so sklearn reports `df = 1`
/// for a degenerate single-class problem where the class count would say `0`.
/// That case is reproduced by using the INDICATOR row count, not `n_classes`.
pub fn chi2(x: &[f64], y: &[f64], n: usize, d: usize) -> Result<ScoreResult, AlgoError> {
    if let Some(neg) = x.iter().find(|v| **v < 0.0) {
        return Err(AlgoError::InvalidFeatureInput {
            estimator: "chi2",
            // sklearn's own wording, verbatim, so a caller matching on the
            // message (including sklearn's estimator_checks) sees what it
            // expects.
            reason: format!("Input X must be non-negative. (found {neg})"),
        });
    }
    let (labels, classes) = class_indices(y);
    let moments = class_col_sums(x, &labels, n, d, classes.len())?;
    // `LabelBinarizer` collapses a <=2-class target to a single indicator column
    // and sklearn appends `1 − Y`, so the observed table has 2 rows there
    // regardless. For 3+ classes the indicator has one row per class.
    let rows = if classes.len() <= 2 { 2 } else { classes.len() };
    let n_f = n as f64;

    let mut scores = Vec::with_capacity(d);
    let mut pvalues = Vec::with_capacity(d);
    for c in 0..d {
        let feature_count = moments.total_sum[c];
        let mut chisq = 0.0;
        for k in 0..rows {
            // With a single class, class 0's observed row is the whole column
            // and the appended `1 − Y` row is identically zero.
            let observed = if k < classes.len() {
                moments.sums[k * d + c]
            } else {
                0.0
            };
            let count_k = if k < classes.len() {
                moments.counts[k]
            } else {
                0
            };
            let class_prob = count_k as f64 / n_f;
            let expected = class_prob * feature_count;
            let diff = observed - expected;
            // sklearn divides under `np.errstate(invalid="ignore")`: an
            // all-zero column makes `expected == 0` and the term `0/0 = NaN`,
            // which propagates through the sum. Reproduced deliberately.
            chisq += diff * diff / expected;
        }
        scores.push(chisq);
        pvalues.push(chi2_sf(chisq, (rows - 1) as f64));
    }
    Ok(ScoreResult::with_pvalues(scores, pvalues))
}

// ===========================================================================
// r_regression / f_regression
// ===========================================================================

/// `sklearn.feature_selection.r_regression(X, y, center=, force_finite=)` —
/// Pearson's r between each column and the target.
///
/// sklearn evaluates `⟨y − ȳ, X⟩ / (‖x − x̄‖ · ‖y − ȳ‖)`, exploiting
/// `E[(x − x̄)(y − ȳ)] = E[x(y − ȳ)]` so `X` is never materially centered — only
/// its NORM is corrected, via the moment identity `‖x − x̄‖² = Σx² − n·x̄²`.
///
/// ## Which parts follow sklearn's ARITHMETIC and not merely its algebra
/// Two choices here are about rounding, not about the formula, and both were
/// forced by the oracle:
///
/// * **`y` IS materially centered**, and the covariance is the literal dot
///   product `Σ (yᵢ − ȳ)·xᵢ` — not the algebraically-equal `Σy·x − ȳ·Σx`. On a
///   CONSTANT column the two differ qualitatively: the dot product leaves a tiny
///   non-zero residue where the moment form cancels to exactly `0.0`, and since
///   the denominator is also `0` there, one route gives `±inf` and the other
///   `NaN` — which `force_finite` then maps to `0.0`. sklearn returns the `±inf`
///   (its `force_finite` maps only `NaN`), so reproducing its answer on a
///   constant column requires reproducing its subtraction order.
/// * **`‖x − x̄‖²` is NOT clamped at zero** before the square root. sklearn takes
///   `np.sqrt` of the raw moment difference, which for a constant column can be a
///   small NEGATIVE number and yields `NaN`. Clamping would turn that into a `0`
///   denominator and change which of the two degenerate branches above is taken.
///
/// The moment identity is otherwise kept as-is (rather than swapped for the
/// numerically stabler two-pass form) for the same reason: the comparison is
/// against sklearn, and the two forms disagree only on a column whose variance is
/// negligible against its mean — where sklearn's answer is the one the user is
/// migrating from.
///
/// With `center = false` the norms are the RAW `‖x‖`, `‖y‖` and no mean is
/// removed anywhere — sklearn's documented "already centered" fast path.
pub fn r_regression(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    center: bool,
    force_finite: bool,
) -> Result<Vec<f64>, AlgoError> {
    let n_f = n as f64;
    // Center `y` up front so the sweep's `xy` accumulator IS sklearn's
    // `safe_sparse_dot(y, X)` and its `ysq` IS `np.linalg.norm(y)²` (see above).
    let y_work: Vec<f64> = if center {
        let mean = y.iter().sum::<f64>() / n_f;
        y.iter().map(|v| v - mean).collect()
    } else {
        y.to_vec()
    };
    let m = cross_moments(x, &y_work, n, d)?;
    let y_norm = m.ysq.sqrt();

    let mut out = Vec::with_capacity(d);
    for c in 0..d {
        let x_norm = if center {
            let x_mean = m.xsum[c] / n_f;
            (m.xsq[c] - n_f * x_mean * x_mean).sqrt()
        } else {
            m.xsq[c].sqrt()
        };
        let mut r = m.xy[c] / x_norm / y_norm;
        if force_finite && r.is_nan() {
            // Constant column or constant target: sklearn sets the correlation
            // to its minimum, 0.0. It maps ONLY `NaN` — an infinite `r` from a
            // zero norm with a non-zero residual dot product survives, which is
            // why the two arithmetic choices documented above matter.
            r = 0.0;
        }
        out.push(r);
    }
    Ok(out)
}

/// `sklearn.feature_selection.f_regression(X, y, center=, force_finite=)` — the
/// univariate linear-regression F-statistic of each column and its p-value.
///
/// `F = r²/(1 − r²) · dof` with `dof = n − 2` when centering (`n − 1` when not),
/// and `p = fdtrc(1, dof, F)`. The `force_finite = true` default then rewrites
/// the two degenerate outcomes, and the exact sentinels matter to a caller
/// comparing `scores_`:
///
/// * a perfectly (anti-)correlated column gives `r² = 1`, so `F = ±inf` → sklearn
///   writes `np.finfo(dtype).max` (i.e. [`f64::MAX`]) with p-value `0.0`;
/// * a constant column or target gives `r = 0` under `force_finite`, whose
///   `F` is a clean `0.0` — but with `force_finite = false` the `r` is `NaN`, so
///   `F` is `NaN` and sklearn writes `0.0` for the score and `1.0` for the
///   p-value ONLY when `force_finite` is on.
///
/// Note the ORDER sklearn applies these in: `r_regression` is called with the
/// same `force_finite`, so with it ON the `NaN` r has already become `0` and the
/// `mask_nan` branch is unreachable from a constant column; it remains reachable
/// from a `NaN` that arises in the `r²/(1 − r²)` step itself. Reproducing the
/// order rather than the intent is what keeps the two flags independent.
pub fn f_regression(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    center: bool,
    force_finite: bool,
) -> Result<ScoreResult, AlgoError> {
    let r = r_regression(x, y, n, d, center, force_finite)?;
    let dof = (n - if center { 2 } else { 1 }) as f64;
    let mut scores = Vec::with_capacity(d);
    let mut pvalues = Vec::with_capacity(d);
    for &ri in r.iter() {
        let r2 = ri * ri;
        let mut f = r2 / (1.0 - r2) * dof;
        let mut p = f_sf(f, 1.0, dof);
        if force_finite {
            if f.is_infinite() {
                f = f64::MAX;
                p = 0.0;
            } else if f.is_nan() {
                f = 0.0;
                p = 1.0;
            }
        }
        scores.push(f);
        pvalues.push(p);
    }
    Ok(ScoreResult::with_pvalues(scores, pvalues))
}
