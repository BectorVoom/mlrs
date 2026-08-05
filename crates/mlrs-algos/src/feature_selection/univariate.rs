//! The six UNIVARIATE filters (FSEL-01) — `SelectKBest`, `SelectPercentile`,
//! `SelectFpr`, `SelectFdr`, `SelectFwe`, `GenericUnivariateSelect`.
//!
//! All six share sklearn's `_BaseFilter`: `fit` runs `score_func(X, y)`, stores
//! `scores_` and (when produced) `pvalues_`, and the SUBCLASS supplies only the
//! rule turning those into a mask. mlrs keeps that factoring exactly —
//! [`UnivariateFilter`] is the one estimator type, [`SelectionMode`] is the rule,
//! and the six sklearn class names are constructors returning it.
//!
//! ## Why one type with a mode, and not six types
//! It is what sklearn itself does under the surface, and doing otherwise would
//! be a worse copy. `GenericUnivariateSelect` is DEFINED as "instantiate the
//! class named by `mode`, copy `scores_`/`pvalues_` into it, and ask it for its
//! mask" (`_make_selector` + `_get_support_mask`), so the modes have to be
//! interchangeable at runtime anyway; six structs would need a seventh holding a
//! `Box<dyn>` of the other six to express that. One type with a
//! [`SelectionMode`] enum gives `GenericUnivariateSelect` for free and keeps the
//! five specific constructors as thin, correctly-defaulted entry points.
//!
//! ## The five mask rules, and the details that decide real masks
//!
//! * **`k_best`** — `argsort(clean_nans(scores), kind="mergesort")[-k:]`. The
//!   STABLE sort is specified in sklearn's source ("Request a stable sort") and
//!   is what makes a tie resolve toward the HIGHER column index: `argsort`
//!   ascending puts earlier-indexed ties first, and taking the last `k` therefore
//!   keeps the later ones. `k = 0` selects nothing; `k = "all"` selects
//!   everything; `k > n_features` warns and keeps everything.
//! * **`percentile`** — `threshold = np.percentile(scores, 100 − p)`, mask is
//!   `scores > threshold` (STRICTLY greater), and then ties AT the threshold are
//!   added back in ASCENDING COLUMN ORDER until `int(n_features · p / 100)`
//!   features are selected. `p == 100` and `p == 0` short-circuit BEFORE the
//!   NaN cleaning, so a matrix of all-`NaN` scores still returns all/no features.
//! * **`fpr`** — `pvalues < alpha`, strictly.
//! * **`fdr`** — Benjamini-Hochberg: sort the p-values, keep those satisfying
//!   `p_(i) <= alpha·i/n_features` (1-based `i`), take the LARGEST such `p`, and
//!   select every feature with `p <= that`. If none qualify, select nothing.
//!   Note the final comparison is `<=` while `fpr`/`fwe` use `<`.
//! * **`fwe`** — `pvalues < alpha / n_features` (Bonferroni), strictly.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

use super::score::{ScoreFunc, ScoreResult};
use super::selector::{
    clean_nans, inverse_transform_selected, percentile_linear, transform_selected, Selector,
};

/// `SelectKBest(k=..)`'s parameter: an integer count or sklearn's `"all"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KBest {
    /// `k = <int>`.
    Count(usize),
    /// `k = "all"` — "bypasses selection, for use in a parameter search".
    All,
}

/// Which rule turns `(scores_, pvalues_)` into a support mask — sklearn's
/// `GenericUnivariateSelect(mode=..)`, and equivalently the choice of which of
/// the five specific classes is in use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionMode {
    /// `SelectPercentile(percentile=..)` / `mode='percentile'`. sklearn default
    /// `percentile = 10`.
    Percentile(f64),
    /// `SelectKBest(k=..)` / `mode='k_best'`. sklearn default `k = 10`.
    KBest(KBest),
    /// `SelectFpr(alpha=..)` / `mode='fpr'`. sklearn default `alpha = 0.05`.
    Fpr(f64),
    /// `SelectFdr(alpha=..)` / `mode='fdr'`. sklearn default `alpha = 0.05`.
    Fdr(f64),
    /// `SelectFwe(alpha=..)` / `mode='fwe'`. sklearn default `alpha = 0.05`.
    Fwe(f64),
}

impl SelectionMode {
    /// The sklearn `mode` string, for error messages and `get_params()`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Percentile(_) => "percentile",
            Self::KBest(_) => "k_best",
            Self::Fpr(_) => "fpr",
            Self::Fdr(_) => "fdr",
            Self::Fwe(_) => "fwe",
        }
    }

    /// Parse a `GenericUnivariateSelect(mode=..)` string against a `param`
    /// value, producing the mode the string names.
    ///
    /// `param` carries sklearn's dual typing: it is the `percentile` for
    /// `'percentile'`, the `k` for `'k_best'`, and the `alpha` for the other
    /// three, which is why `GenericUnivariateSelect` has one `param` rather than
    /// three named ones. `"all"` is accepted for `param` and is meaningful only
    /// in `k_best` mode (sklearn's `StrOptions({"all"})` allows it for any mode,
    /// and `_make_selector`'s `set_params` then hands `"all"` to whichever
    /// parameter the mode has — for `percentile` that yields a `TypeError` from
    /// numpy at mask time; mlrs rejects it up front with a message that says
    /// which mode it was, which is strictly more useful and cannot change a
    /// working call's behaviour).
    pub fn from_mode(mode: &str, param: GenericParam) -> Result<Self, AlgoError> {
        let numeric = |p: GenericParam, what: &'static str| -> Result<f64, AlgoError> {
            match p {
                GenericParam::Value(v) => Ok(v),
                GenericParam::All => Err(AlgoError::InvalidSelectorParam {
                    estimator: "generic_univariate_select",
                    param: "param",
                    value: f64::NAN,
                    reason: what,
                }),
            }
        };
        match mode {
            "percentile" => Ok(Self::Percentile(numeric(
                param,
                "'all' is only meaningful in mode='k_best'; percentile needs a number in [0, 100]",
            )?)),
            "k_best" => Ok(Self::KBest(match param {
                GenericParam::All => KBest::All,
                GenericParam::Value(v) => KBest::Count(v as usize),
            })),
            "fpr" => Ok(Self::Fpr(numeric(
                param,
                "'all' is only meaningful in mode='k_best'; fpr needs an alpha in [0, 1]",
            )?)),
            "fdr" => Ok(Self::Fdr(numeric(
                param,
                "'all' is only meaningful in mode='k_best'; fdr needs an alpha in [0, 1]",
            )?)),
            "fwe" => Ok(Self::Fwe(numeric(
                param,
                "'all' is only meaningful in mode='k_best'; fwe needs an alpha in [0, 1]",
            )?)),
            other => Err(AlgoError::UnknownSelectorOption {
                estimator: "generic_univariate_select",
                param: "mode",
                value: other.to_string(),
                expected: "percentile, k_best, fpr, fdr, fwe",
            }),
        }
    }

    /// Validate the data-INDEPENDENT half of this mode's domain, at `build()`.
    ///
    /// The `k > n_features` case is deliberately NOT here: sklearn WARNS about it
    /// at `fit` ("All the features will be returned") rather than raising, so it
    /// is not a rejection at all — see [`UnivariateFilter::k_exceeds_features`].
    fn validate(&self, estimator: &'static str) -> Result<(), AlgoError> {
        let interval = |v: f64, param: &'static str, hi: f64, reason| {
            if v.is_nan() || v < 0.0 || v > hi {
                Err(AlgoError::InvalidSelectorParam {
                    estimator,
                    param,
                    value: v,
                    reason,
                })
            } else {
                Ok(())
            }
        };
        match self {
            Self::Percentile(p) => interval(*p, "percentile", 100.0, "must be in [0, 100]"),
            Self::KBest(_) => Ok(()),
            Self::Fpr(a) | Self::Fdr(a) | Self::Fwe(a) => {
                interval(*a, "alpha", 1.0, "must be in [0, 1]")
            }
        }
    }

    /// Whether this mode thresholds `pvalues_` and therefore needs them.
    fn needs_pvalues(&self) -> bool {
        matches!(self, Self::Fpr(_) | Self::Fdr(_) | Self::Fwe(_))
    }

    /// Apply the rule (module docs) to produce the support mask.
    fn mask(&self, res: &ScoreResult, estimator: &'static str) -> Result<Vec<bool>, AlgoError> {
        let d = res.scores.len();
        match self {
            Self::Percentile(p) => Ok(percentile_mask(&res.scores, *p, d)),
            Self::KBest(k) => Ok(kbest_mask(&res.scores, *k, d)),
            Self::Fpr(alpha) => {
                let pv = pvalues(res, estimator, "fpr")?;
                Ok(pv.iter().map(|&p| p < *alpha).collect())
            }
            Self::Fwe(alpha) => {
                let pv = pvalues(res, estimator, "fwe")?;
                let cut = *alpha / d as f64;
                Ok(pv.iter().map(|&p| p < cut).collect())
            }
            Self::Fdr(alpha) => {
                let pv = pvalues(res, estimator, "fdr")?;
                Ok(fdr_mask(pv, *alpha))
            }
        }
    }
}

/// Fetch `pvalues_` or report the mode/score-function mismatch.
fn pvalues<'a>(
    res: &'a ScoreResult,
    estimator: &'static str,
    mode: &'static str,
) -> Result<&'a [f64], AlgoError> {
    res.pvalues
        .as_deref()
        .ok_or(AlgoError::ScoreFuncHasNoPValues { estimator, mode })
}

/// `SelectKBest._get_support_mask`.
fn kbest_mask(scores: &[f64], k: KBest, d: usize) -> Vec<bool> {
    match k {
        KBest::All => vec![true; d],
        KBest::Count(0) => vec![false; d],
        KBest::Count(k) => {
            let cleaned = clean_nans(scores);
            let mut order: Vec<usize> = (0..d).collect();
            // STABLE ascending sort — sklearn's `kind="mergesort"`. `sort_by` is
            // Rust's stable sort, so equal scores keep ascending index order and
            // taking the last `k` keeps the HIGHER indices of a tie, exactly as
            // sklearn's `argsort(...)[-k:]` does.
            order.sort_by(|&a, &b| cleaned[a].total_cmp(&cleaned[b]));
            let mut mask = vec![false; d];
            for &i in order.iter().skip(d.saturating_sub(k)) {
                mask[i] = true;
            }
            mask
        }
    }
}

/// `SelectPercentile._get_support_mask`.
fn percentile_mask(scores: &[f64], p: f64, d: usize) -> Vec<bool> {
    // sklearn short-circuits BOTH endpoints before `_clean_nans` ("Cater for
    // NaNs"), so an all-NaN score vector still returns all / no features rather
    // than whatever the percentile of `f64::MIN`s would give.
    if p == 100.0 {
        return vec![true; d];
    }
    if p == 0.0 {
        return vec![false; d];
    }
    let cleaned = clean_nans(scores);
    let threshold = percentile_linear(&cleaned, 100.0 - p);
    let mut mask: Vec<bool> = cleaned.iter().map(|&s| s > threshold).collect();
    // Tie handling: features exactly AT the threshold are added back, in
    // ascending column order, up to `int(d · p / 100)` total. `max_feats` can be
    // BELOW the already-selected count (when many scores exceed the threshold),
    // in which case `max_feats - selected` underflows in Python to a negative
    // slice bound and `ties[:negative]` selects a PREFIX — sklearn's own
    // behaviour, which drops the last `|negative|` ties instead of adding any.
    // Reproduced via a signed budget rather than a `usize` subtraction, which
    // would panic.
    let ties: Vec<usize> = (0..d).filter(|&i| cleaned[i] == threshold).collect();
    if !ties.is_empty() {
        let max_feats = (d as f64 * p / 100.0) as isize;
        let selected = mask.iter().filter(|&&m| m).count() as isize;
        let budget = max_feats - selected;
        let take = if budget >= 0 {
            (budget as usize).min(ties.len())
        } else {
            // Python's `ties[:negative]` keeps `len + negative` items.
            (ties.len() as isize + budget).max(0) as usize
        };
        for &i in ties.iter().take(take) {
            mask[i] = true;
        }
    }
    mask
}

/// `SelectFdr._get_support_mask` — Benjamini-Hochberg.
fn fdr_mask(pv: &[f64], alpha: f64) -> Vec<bool> {
    let d = pv.len();
    let mut sorted: Vec<f64> = pv.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut best: Option<f64> = None;
    for (i, &p) in sorted.iter().enumerate() {
        if p <= alpha / d as f64 * (i + 1) as f64 {
            // `selected.max()` over ALL qualifying entries, not the last
            // qualifying prefix: sklearn filters the whole sorted vector and
            // takes the maximum, so a non-monotone qualification pattern
            // (possible, since the bound grows with `i` while `p` also grows)
            // keeps the largest qualifying p-value rather than stopping at the
            // first failure.
            best = Some(match best {
                Some(b) if b > p => b,
                _ => p,
            });
        }
    }
    match best {
        None => vec![false; d],
        Some(cut) => pv.iter().map(|&p| p <= cut).collect(),
    }
}

/// `GenericUnivariateSelect(param=..)`'s dual-typed parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GenericParam {
    /// A number, interpreted per the mode (percentile / k / alpha).
    Value(f64),
    /// sklearn's `"all"`.
    All,
}

/// The single univariate-filter estimator behind all six sklearn classes.
#[derive(Debug, Clone)]
pub struct UnivariateFilter<F, S = Unfit> {
    score_func: ScoreFunc,
    mode: SelectionMode,
    /// Which sklearn class name this instance stands for, for error messages.
    estimator: &'static str,
    scores: Vec<f64>,
    pvalues: Option<Vec<f64>>,
    support: Vec<bool>,
    _state: PhantomData<(F, S)>,
}

impl<F> UnivariateFilter<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn build(
        score_func: ScoreFunc,
        mode: SelectionMode,
        estimator: &'static str,
    ) -> Result<Self, AlgoError> {
        mode.validate(estimator)?;
        Ok(Self {
            score_func,
            mode,
            estimator,
            scores: Vec::new(),
            pvalues: None,
            support: Vec::new(),
            _state: PhantomData,
        })
    }

    /// `sklearn.feature_selection.SelectKBest(score_func=f_classif, k=10)`.
    pub fn k_best(score_func: ScoreFunc, k: KBest) -> Result<Self, AlgoError> {
        Self::build(score_func, SelectionMode::KBest(k), "select_k_best")
    }

    /// `sklearn.feature_selection.SelectPercentile(score_func=f_classif,
    /// percentile=10)`.
    pub fn percentile(score_func: ScoreFunc, percentile: f64) -> Result<Self, AlgoError> {
        Self::build(
            score_func,
            SelectionMode::Percentile(percentile),
            "select_percentile",
        )
    }

    /// `sklearn.feature_selection.SelectFpr(score_func=f_classif, alpha=5e-2)`.
    pub fn fpr(score_func: ScoreFunc, alpha: f64) -> Result<Self, AlgoError> {
        Self::build(score_func, SelectionMode::Fpr(alpha), "select_fpr")
    }

    /// `sklearn.feature_selection.SelectFdr(score_func=f_classif, alpha=5e-2)`.
    pub fn fdr(score_func: ScoreFunc, alpha: f64) -> Result<Self, AlgoError> {
        Self::build(score_func, SelectionMode::Fdr(alpha), "select_fdr")
    }

    /// `sklearn.feature_selection.SelectFwe(score_func=f_classif, alpha=5e-2)`.
    pub fn fwe(score_func: ScoreFunc, alpha: f64) -> Result<Self, AlgoError> {
        Self::build(score_func, SelectionMode::Fwe(alpha), "select_fwe")
    }

    /// `sklearn.feature_selection.GenericUnivariateSelect(score_func=f_classif,
    /// mode='percentile', param=1e-5)`.
    pub fn generic(
        score_func: ScoreFunc,
        mode: &str,
        param: GenericParam,
    ) -> Result<Self, AlgoError> {
        let mode = SelectionMode::from_mode(mode, param)?;
        Self::build(score_func, mode, "generic_univariate_select")
    }
}

impl<F, S> UnivariateFilter<F, S> {
    /// Whether `fit` would warn that `k` exceeds `n_features` (sklearn's
    /// `_check_params` warning, "All the features will be returned").
    ///
    /// Exposed rather than logged so the PyO3 layer can raise the Python
    /// `UserWarning` sklearn raises — a Rust `log::warn!` would not reach a
    /// `pytest.warns` assertion, and sklearn's own tests check for it.
    pub fn k_exceeds_features(&self, n_features: usize) -> bool {
        matches!(self.mode, SelectionMode::KBest(KBest::Count(k)) if k > n_features)
    }

    /// The selection mode in use.
    pub fn mode(&self) -> SelectionMode {
        self.mode
    }
}

impl<F> Fit<F> for UnivariateFilter<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = UnivariateFilter<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let (n, d) = shape;
        // sklearn's `_BaseFilter.fit` accepts `y=None` and passes it straight to
        // `score_func`; `SelectKBest`/`SelectPercentile` set
        // `target_tags.required = False` to advertise that an unsupervised score
        // function is allowed. None of the BUILT-IN score functions work without
        // `y`, so a missing target is rejected here for them and accepted for a
        // `Custom` one, which is the only case where it can be meaningful.
        let y_host: Vec<f64> = match y {
            Some(yd) => {
                if yd.len() != n {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "y",
                        rows: n,
                        cols: 1,
                        len: yd.len(),
                    }));
                }
                yd.to_host(pool).into_iter().map(host_to_f64).collect()
            }
            None => {
                if !matches!(self.score_func, ScoreFunc::Custom(_)) {
                    return Err(AlgoError::InvalidLabels {
                        estimator: self.estimator,
                        reason: format!(
                            "score function {:?} is supervised and requires y",
                            self.score_func
                        ),
                    });
                }
                Vec::new()
            }
        };
        let x_host: Vec<f64> = x.to_host(pool).into_iter().map(host_to_f64).collect();

        // Reject a p-value-less score function BEFORE running it when the mode
        // is statically known to need p-values: an expensive
        // `mutual_info_regression` sweep should not run only to be discarded.
        // A `Custom` function still has to run first — its output shape is not
        // knowable in advance — and is checked below.
        if self.mode.needs_pvalues() && !self.score_func.yields_pvalues() {
            return Err(AlgoError::ScoreFuncHasNoPValues {
                estimator: self.estimator,
                mode: self.mode.name(),
            });
        }

        let res = self.score_func.eval(&x_host, &y_host, n, d)?;
        if res.scores.len() != d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "scores",
                rows: 1,
                cols: d,
                len: res.scores.len(),
            }));
        }
        if let Some(pv) = res.pvalues.as_ref() {
            if pv.len() != d {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "pvalues",
                    rows: 1,
                    cols: d,
                    len: pv.len(),
                }));
            }
        }
        let support = self.mode.mask(&res, self.estimator)?;

        Ok(UnivariateFilter {
            score_func: self.score_func,
            mode: self.mode,
            estimator: self.estimator,
            scores: res.scores,
            pvalues: res.pvalues,
            support,
            _state: PhantomData,
        })
    }
}

impl<F> UnivariateFilter<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `scores_` — one score per input feature, `NaN`s INTACT (see
    /// [`clean_nans`]).
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// `pvalues_` — `None` when the score function returned scores only.
    pub fn pvalues(&self) -> Option<&[f64]> {
        self.pvalues.as_deref()
    }
}

impl<F, S> Selector for UnivariateFilter<F, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F> Transform<F> for UnivariateFilter<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, self.estimator)
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}
