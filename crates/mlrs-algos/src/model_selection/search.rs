//! Hyper-parameter search drivers (MODSEL-RS-06).
//!
//! The estimator call is the one thing Rust cannot own, so every driver here
//! takes an **evaluator closure**: `(candidate, split) -> score`. Everything
//! around it — which candidates to try, in what order, against which splits,
//! how to reduce the scores, which candidate wins, and (for successive
//! halving) how many resources each round gets and who survives it — is this
//! module's.
//!
//! That split is what lets one implementation serve two callers: the Python
//! shim passes a closure that calls back into `sklearn`-style `fit`/`score`,
//! while a pure-Rust caller passes a closure that fits an `mlrs_algos`
//! estimator. Neither one re-derives the schedule.

use super::validate::{summarize_scores, ScoreSummary};
use super::{value_err, Result};

/// The outcome of a search: the reduced scores plus the winning candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    /// Candidate indices actually evaluated, in evaluation order.
    pub candidates: Vec<usize>,
    /// Row-major `(n_candidates, n_splits)` raw scores.
    pub scores: Vec<f64>,
    /// The per-candidate reduction (`mean` / `std` / `rank`).
    pub summary: ScoreSummary,
    /// Index INTO [`candidates`](Self::candidates) of the highest mean score.
    pub best: usize,
}

/// Evaluate every candidate against every split and rank them
/// (`GridSearchCV` / `RandomizedSearchCV`'s `_run_search`).
///
/// `evaluate(candidate, split)` returns the test score for that pair; return
/// `f64::NAN` to model sklearn's `error_score=np.nan` for a fit that failed.
///
/// The evaluation order is candidate-major, matching sklearn's
/// `product(candidate_params, cv.split(...))` — which matters whenever the
/// closure itself consumes a random stream.
pub fn evaluate_candidates<F>(
    candidates: &[usize],
    n_splits: usize,
    mut evaluate: F,
) -> Result<SearchResults>
where
    F: FnMut(usize, usize) -> f64,
{
    if candidates.is_empty() {
        return Err(value_err!(
            "No fits were performed. Was the CV iterator empty?"
        ));
    }
    if n_splits == 0 {
        return Err(value_err!("Cannot perform a search with 0 splits."));
    }
    let mut scores = Vec::with_capacity(candidates.len() * n_splits);
    for &candidate in candidates {
        for split in 0..n_splits {
            scores.push(evaluate(candidate, split));
        }
    }
    let summary = summarize_scores(&scores, candidates.len(), n_splits)?;
    let best = best_index(&summary.mean);
    Ok(SearchResults {
        candidates: candidates.to_vec(),
        scores,
        summary,
        best,
    })
}

/// The index of the best mean score — `np.argmax` semantics with sklearn's
/// NaN handling (a NaN never wins; the FIRST maximum wins a tie).
pub fn best_index(mean_scores: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (i, &v) in mean_scores.iter().enumerate() {
        if !v.is_nan() && v > best_value {
            best_value = v;
            best = i;
        }
    }
    best
}

/// `_top_k(results, k, itr)` — the `k` best candidates of a halving round.
///
/// sklearn sorts ascending and takes the last `k`, having first rolled the
/// NaNs to the FRONT so they are dropped first. The returned order is the
/// argsort order (worst-to-best among the survivors), which is the order the
/// next round evaluates in — so it is reproduced rather than normalized.
pub fn top_k(mean_scores: &[f64], k: usize) -> Vec<usize> {
    let n_nan = mean_scores.iter().filter(|v| v.is_nan()).count();
    // `np.argsort` puts NaNs at the END; `np.roll(..., n_nan)` then moves them
    // to the front, leaving the finite scores ascending in the tail.
    let mut order: Vec<usize> = (0..mean_scores.len()).collect();
    order.sort_by(
        |&a, &b| match (mean_scores[a].is_nan(), mean_scores[b].is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => mean_scores[a]
                .partial_cmp(&mean_scores[b])
                .expect("both finite")
                .then(a.cmp(&b)),
        },
    );
    order.rotate_right(n_nan);
    let k = k.min(order.len());
    order[order.len() - k..].to_vec()
}

/// Where a successive-halving run gets its `min_resources_` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinResources {
    /// `"smallest"` — `n_splits * 2`, times `n_classes` for a classifier.
    Smallest,
    /// `"exhaust"` — raised so the LAST round uses as much as possible.
    Exhaust,
    /// An explicit row count.
    Fixed(usize),
}

/// The parameters of a successive-halving run, before the schedule is derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalvingParams {
    pub factor: usize,
    pub min_resources: MinResources,
    pub max_resources: usize,
    pub aggressive_elimination: bool,
    /// `n_splits * 2 * n_classes` for `MinResources::Smallest`; the caller
    /// computes it because only it knows whether the estimator is a classifier.
    pub smallest_resources: usize,
}

/// One round of a halving run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalvingIteration {
    pub iteration: usize,
    /// Rows (or `resource` units) each candidate is fit on this round.
    pub n_resources: usize,
    /// Candidates entering this round.
    pub n_candidates: usize,
}

/// The full derived schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HalvingSchedule {
    pub min_resources: usize,
    pub max_resources: usize,
    pub n_required_iterations: usize,
    pub n_possible_iterations: usize,
    pub iterations: Vec<HalvingIteration>,
}

/// Derive the successive-halving schedule (`BaseSuccessiveHalving._run_search`).
///
/// Two counts drive everything:
///
/// * `n_required_iterations = 1 + floor(log_factor(n_candidates))` — how many
///   rounds it takes to eliminate down to one candidate;
/// * `n_possible_iterations = 1 + floor(log_factor(max/min))` — how many rounds
///   the resource budget allows.
///
/// Without `aggressive_elimination` the run stops at the smaller of the two, so
/// it may finish with several candidates still standing. WITH it, the early
/// rounds are *pinned* to the starting resource level for as long as it takes
/// to catch up, so the run always reaches a single candidate — which is why
/// `power` is offset rather than simply being the iteration number.
pub fn halving_schedule(params: HalvingParams, n_candidates: usize) -> Result<HalvingSchedule> {
    if params.factor < 2 {
        return Err(value_err!("factor must be >= 2, got {}", params.factor));
    }
    if n_candidates == 0 {
        return Err(value_err!("No candidates to search."));
    }

    let factor = params.factor as f64;
    let n_required_iterations = 1 + (n_candidates as f64).log(factor).floor() as usize;

    let mut min_resources = match params.min_resources {
        MinResources::Fixed(v) => v,
        MinResources::Smallest | MinResources::Exhaust => params.smallest_resources.max(1),
    };
    if matches!(params.min_resources, MinResources::Exhaust) {
        // Raise `min_resources` so the LAST required round lands on
        // `max_resources` — "exhaust" means "do not leave budget unused".
        let last_iteration = n_required_iterations - 1;
        let divisor = params.factor.pow(last_iteration as u32);
        min_resources = min_resources.max(params.max_resources / divisor.max(1));
    }
    if min_resources == 0 {
        return Err(value_err!(
            "min_resources_=0: you might have passed an empty dataset X."
        ));
    }
    if min_resources > params.max_resources {
        return Err(value_err!(
            "min_resources_={min_resources} is greater than max_resources_={}.",
            params.max_resources
        ));
    }

    let n_possible_iterations = 1
        + ((params.max_resources / min_resources) as f64)
            .log(factor)
            .floor() as usize;
    let n_iterations = if params.aggressive_elimination {
        n_required_iterations
    } else {
        n_possible_iterations.min(n_required_iterations)
    };

    let mut iterations = Vec::with_capacity(n_iterations);
    let mut remaining = n_candidates;
    for itr in 0..n_iterations {
        let power = if params.aggressive_elimination {
            // Clamp at 0 so the first rounds repeat the starting resource
            // level while the candidate pool catches up.
            (itr + n_possible_iterations).saturating_sub(n_required_iterations)
        } else {
            itr
        };
        let n_resources = (params
            .factor
            .saturating_pow(power as u32)
            .saturating_mul(min_resources))
        .min(params.max_resources);
        iterations.push(HalvingIteration {
            iteration: itr,
            n_resources,
            n_candidates: remaining,
        });
        remaining = remaining.div_ceil(params.factor);
    }

    Ok(HalvingSchedule {
        min_resources,
        max_resources: params.max_resources,
        n_required_iterations,
        n_possible_iterations,
        iterations,
    })
}

/// `HalvingRandomSearchCV`'s `n_candidates="exhaust"` — draw enough candidates
/// that the LAST round can use the whole resource budget.
pub fn exhaust_n_candidates(max_resources: usize, min_resources: usize) -> usize {
    if min_resources == 0 {
        1
    } else {
        (max_resources / min_resources).max(1)
    }
}

/// Run a full successive-halving search over a closure evaluator.
///
/// `evaluate(candidate, split, n_resources)` scores one candidate on one split
/// at the round's resource level. Returns the per-round results, newest last;
/// sklearn's `best_estimator_` comes from the LAST round, not from the best
/// score across all rounds — a candidate that scored well on 20 rows has not
/// earned a win over one measured on 2000.
pub fn run_halving<F>(
    params: HalvingParams,
    n_candidates: usize,
    n_splits: usize,
    mut evaluate: F,
) -> Result<Vec<SearchResults>>
where
    F: FnMut(usize, usize, usize) -> f64,
{
    let schedule = halving_schedule(params, n_candidates)?;
    let mut candidates: Vec<usize> = (0..n_candidates).collect();
    let mut rounds = Vec::with_capacity(schedule.iterations.len());

    for iteration in &schedule.iterations {
        let n_resources = iteration.n_resources;
        let results = evaluate_candidates(&candidates, n_splits, |candidate, split| {
            evaluate(candidate, split, n_resources)
        })?;
        let keep = candidates.len().div_ceil(params.factor);
        let survivors = top_k(&results.summary.mean, keep);
        candidates = survivors
            .into_iter()
            .map(|i| results.candidates[i])
            .collect();
        rounds.push(results);
    }
    Ok(rounds)
}
