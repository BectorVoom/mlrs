//! `voting` — the structural core of the prediction-voting meta-estimator
//! (VOTE-01), behind `mlrs.ensemble.VotingRegressor`.
//!
//! Voting is a *composition*, like [`crate::ensemble::stacking`]: the arithmetic
//! that dominates a fit already runs on the device inside the composed members.
//! What the meta-estimator itself owns is
//!
//! 1. **structure** — which entries are dropped, which weights survive that
//!    drop, and what the transform columns are called; and
//! 2. **one small aggregation** — the `n × k` column stack (`transform`) and the
//!    weighted row mean (`predict`).
//!
//! The structural half that voting shares with stacking is not duplicated here:
//! `_BaseComposition._validate_names` and the `'drop'` bookkeeping are
//! [`stacking::validate_names`] and [`stacking::kept_indices`], which are
//! sklearn's own shared base-class rules and are called by both shims. What IS
//! here is everything voting does differently.
//!
//! ## The aggregation, and why it has a device arm at all
//!
//! [`weighted_average`] is `np.average(mat, axis=1, weights=w)` and
//! [`stack_columns`] is `np.asarray([…]).T`. Both are small, and — exactly as in
//! stacking — `numpy` remains the shipping default because the members' columns
//! are ALREADY host-resident when they arrive. The difference worth stating is
//! that `predict` **reduces**: it consumes `n · k` and produces `n`, so a device
//! arm downloads `k` times less than it uploads and has real arithmetic to
//! amortise the crossing against, which is not true of stacking's pure copy.
//! `docs/voting.md` carries the measured ladder that settles which arm ships;
//! [`VoteEngine`] is the knob.
//!
//! ## sklearn parity
//!
//! Every rule reproduced here is observable as an exception message, a
//! `get_feature_names_out()` string, or a predicted value, so each is matched
//! against `sklearn.ensemble._voting`:
//!
//! - [`check_weights_len`] — `_BaseVoting.fit`'s length check
//! - [`active_weight_slots`] — `_BaseVoting._weights_not_none`
//! - [`transform_feature_names`] — `VotingRegressor.get_feature_names_out`
//! - [`stack_columns`] — `_BaseVoting._predict`
//! - [`weighted_average`] — `VotingRegressor.predict`'s `np.average`
//!
//! Tests live in `crates/mlrs-algos/tests/voting_test.rs` (AGENTS.md §2 — never
//! an in-source `#[cfg(test)] mod tests`).

use std::ops::{Add, Div, Mul};

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use thiserror::Error;

use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::voting::{
    vote_average_device, vote_hard_predict_device, vote_hstack_device, vote_soft_predict_device,
    vote_soft_proba_device, vote_transform_device, VoteEngine,
};
use mlrs_backend::runtime::ActiveRuntime;

/// A voting-composition failure.
///
/// Two variants, because voting has exactly two failure classes that reach the
/// caller with DIFFERENT Python exception types. Everything structural is a
/// `ValueError`, as in [`stacking`](crate::ensemble::stacking) — but a zero
/// weight sum is numpy's `ZeroDivisionError`, raised from inside `np.average`,
/// and collapsing it into `ValueError` would break a caller that migrated its
/// `except` clause over from sklearn.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VotingError {
    /// Maps to `ValueError` at the Python boundary, message verbatim.
    #[error("{0}")]
    Value(String),
    /// Maps to `ZeroDivisionError`; the message is numpy's own.
    #[error("Weights sum to zero, can't be normalized")]
    ZeroWeightSum,
}

type Result<T> = std::result::Result<T, VotingError>;

fn value_err(msg: impl Into<String>) -> VotingError {
    VotingError::Value(msg.into())
}

/// sklearn `_BaseVoting.fit`: `weights` must be as long as `estimators`.
///
/// Note that it is compared against the FULL `estimators` list, not the kept
/// one — a `'drop'`ped entry still needs its weight slot present, and
/// [`active_weight_slots`] removes it afterwards. Getting this backwards would make
/// `set_params(lr='drop')` on a weighted ensemble raise where sklearn succeeds.
///
/// The message is sklearn's, backticks included.
pub fn check_weights_len(n_weights: usize, n_estimators: usize) -> Result<()> {
    if n_weights != n_estimators {
        return Err(value_err(format!(
            "Number of `estimators` and weights must be equal; got \
             {n_weights} weights, {n_estimators} estimators"
        )));
    }
    Ok(())
}

/// sklearn `_BaseVoting._weights_not_none`: the POSITIONS of the weights whose
/// entries survived the `'drop'` filter, in list order.
///
/// `is_drop[i]` is the shim's answer to `estimators[i][1] == 'drop'` — the
/// comparison itself has to happen in Python (the value is an arbitrary
/// estimator object whose `__eq__` may be overloaded), the consequences are
/// here.
///
/// **Positions rather than values, deliberately.** sklearn's own version
/// returns a list of the weight OBJECTS, and numpy then infers the result dtype
/// from them: a `float32` weight array leaves `predict` in `float32` where a
/// Python-float list promotes it to `float64`. Carrying the values through this
/// function as `f64` would silently erase that distinction — so the rule crosses
/// as indices and the shim indexes its own untouched objects, which is the only
/// shape that cannot lose a dtype.
///
/// sklearn `zip`s `estimators` with `weights` and so silently truncates to the
/// shorter of the two; that cannot be observed there because
/// [`check_weights_len`] has already run, and it is rejected here rather than
/// reproduced so a future caller that skips the length check gets an error
/// instead of a quietly short weight vector.
pub fn active_weight_slots(n_weights: usize, is_drop: &[bool]) -> Result<Vec<usize>> {
    check_weights_len(n_weights, is_drop.len())?;
    Ok(is_drop
        .iter()
        .enumerate()
        .filter(|(_, &d)| !d)
        .map(|(i, _)| i)
        .collect())
}

/// sklearn `VotingRegressor.get_feature_names_out`.
///
/// One name per KEPT member, `"{class}_{name}"` — a regressor contributes
/// exactly one transform column, so unlike stacking's
/// `meta_feature_names` there is no within-block index and no `passthrough`
/// tail. `class_name` is the lower-cased class name (`"votingregressor"`),
/// passed in rather than hard-coded so a subclass reports its own.
pub fn transform_feature_names(class_name: &str, kept_names: &[String]) -> Vec<String> {
    kept_names
        .iter()
        .map(|name| format!("{class_name}_{name}"))
        .collect()
}

// ------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// ------------------------------------------------------------------------- //

/// sklearn `VotingClassifier`'s `voting` parameter — the one string-valued
/// scalar on this estimator, and the one that decides which aggregation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Voting {
    /// Weighted majority over the members' LABELS.
    Hard,
    /// Weighted average of the members' `predict_proba` blocks.
    Soft,
}

impl Voting {
    /// The knob spelling, so a report of the mode that ran round-trips.
    pub fn as_str(self) -> &'static str {
        match self {
            Voting::Hard => "hard",
            Voting::Soft => "soft",
        }
    }
}

/// Validate the `voting` constructor argument.
///
/// sklearn declares it as `StrOptions({"hard", "soft"})`, so an unrecognized
/// value is an `InvalidParameterError` — the shim re-raises this `ValueError`
/// under that class, exactly as it does for stacking's `stack_method`.
///
/// **The option ORDER in the message is not part of parity**, for the reason
/// [`crate::ensemble::stacking::stack_method_request`] spells out: sklearn
/// renders a `StrOptions` constraint by iterating a Python `set`, whose order
/// for these two strings changes with `PYTHONHASHSEED`. The oracle test parses
/// the option set out of both messages rather than comparing them as text.
pub fn voting_mode(value: &str) -> Result<Voting> {
    match value {
        "hard" => Ok(Voting::Hard),
        "soft" => Ok(Voting::Soft),
        _ => Err(value_err(format!(
            "The 'voting' parameter of VotingClassifier must be a str among \
             {{'hard', 'soft'}}. Got '{value}' instead."
        ))),
    }
}

/// sklearn `VotingClassifier.get_feature_names_out`.
///
/// The names depend on BOTH string-ish parameters, which is why this cannot be
/// [`transform_feature_names`]:
///
/// * `voting='hard'` — one column per kept member, `"{class}_{name}"`, the same
///   shape the regressor produces;
/// * `voting='soft'` — `n_classes` columns per kept member,
///   `"{class}_{name}{i}"`, member-major, matching `np.hstack(probas)`'s layout.
///   Note there is no separator before `i`: sklearn writes
///   `f"{class_name}_{name}{i}"`, so a member called `lr` on a 3-class problem
///   yields `votingclassifier_lr0/1/2`.
/// * `voting='soft'` with `flatten_transform=False` has no 2-D output to name at
///   all, and sklearn rejects it. That check is
///   [`check_feature_names_supported`], run first by the shim so the rejection
///   precedes the naming rather than depending on it.
pub fn classifier_feature_names(
    class_name: &str,
    kept_names: &[String],
    voting: Voting,
    n_classes: usize,
) -> Vec<String> {
    match voting {
        Voting::Hard => transform_feature_names(class_name, kept_names),
        Voting::Soft => kept_names
            .iter()
            .flat_map(|name| (0..n_classes).map(move |i| format!("{class_name}_{name}{i}")))
            .collect(),
    }
}

/// sklearn `VotingClassifier.get_feature_names_out`'s one rejection.
///
/// `voting='soft', flatten_transform=False` makes `transform` return a 3-D
/// `(k, n, C)` stack, which has no column names, so sklearn raises rather than
/// inventing any. The message is verbatim, backticks included.
pub fn check_feature_names_supported(voting: Voting, flatten_transform: bool) -> Result<()> {
    if voting == Voting::Soft && !flatten_transform {
        return Err(value_err(
            "get_feature_names_out is not supported when `voting='soft'` and \
             `flatten_transform=False`",
        ));
    }
    Ok(())
}

/// sklearn `VotingClassifier.predict` under `voting='hard'` — the weighted
/// majority label of each row.
///
/// This is
/// `np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=w)), 1, preds)`
/// with the Python-level per-row loop gone. `preds[j]` is member `j`'s
/// `n_rows`-long ENCODED prediction column, `n_bins` is one past the largest
/// label any of them produced, and the answer is the argmax INDEX — a position
/// in the caller's class order, which the shim maps back through `classes_`.
///
/// **The tally is `f64` regardless of anything the caller holds**, because
/// `np.bincount`'s `weights` accumulator is `float64` regardless of the weights'
/// own dtype. The uniform case (`weights = None`) is a sum of `1.0`s, which is
/// exact in `f64` for any `k`, and matches numpy's `int64` counting bit for bit.
///
/// **The argmax is bounded by each row's own largest label**, not by `n_bins`:
/// `np.bincount` returns `x.max() + 1` entries, so a class above the row's
/// maximum is not a candidate even when it would win. That is invisible under
/// non-negative weights and decisive under negative ones — see the kernel module
/// docs for the worked example.
///
/// Ties go to the LOWEST index, which is `np.argmax`'s rule.
pub fn hard_vote_labels(
    preds: &[&[u32]],
    weights: Option<&[f64]>,
    n_rows: usize,
    n_bins: usize,
) -> Result<Vec<u32>> {
    check_label_columns(preds, n_rows, n_bins)?;
    let k = preds.len();
    let w: Vec<f64> = match weights {
        Some(w) => {
            if w.len() != k {
                return Err(value_err(format!(
                    "have {k} prediction columns but {} weights",
                    w.len()
                )));
            }
            w.to_vec()
        }
        None => vec![1.0; k],
    };

    // One scratch tally reused across rows, left ZEROED on exit from each row by
    // clearing only the `k` bins that row touched. Clearing the whole `n_bins`
    // per row would be `O(n · n_bins)` for a structure only `k` of whose entries
    // can be non-zero.
    let mut tally = vec![0.0f64; n_bins];
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut hi = 0u32;
        for (j, &pred) in preds.iter().enumerate() {
            let label = pred[r];
            tally[label as usize] += w[j];
            if label > hi {
                hi = label;
            }
        }
        let mut best = tally[0];
        let mut best_idx = 0u32;
        for c in 1..=hi {
            let v = tally[c as usize];
            if v > best {
                best = v;
                best_idx = c;
            }
        }
        out.push(best_idx);
        for &pred in preds {
            tally[pred[r] as usize] = 0.0;
        }
    }
    Ok(out)
}

/// `np.argmax(mat, axis=1)` over a row-major `n_rows × n_cols` matrix.
///
/// The soft route's second half: `predict` is `argmax(predict_proba(X), axis=1)`,
/// and the argmax is separated from the average so `predict_proba` — a public
/// method in its own right — is not computed twice on the host arm.
///
/// FIRST maximum on a tie, which is numpy's rule and the device kernel's.
pub fn argmax_rows<F: Copy + PartialOrd>(
    mat: &[F],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<u32>> {
    if n_cols == 0 {
        return Err(value_err("cannot take an argmax over zero classes"));
    }
    if mat.len() != n_rows * n_cols {
        return Err(value_err(format!(
            "probability matrix has {} elements, expected {n_rows} x {n_cols}",
            mat.len()
        )));
    }
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let base = r * n_cols;
        let mut best = mat[base];
        let mut best_idx = 0u32;
        for c in 1..n_cols {
            if mat[base + c] > best {
                best = mat[base + c];
                best_idx = c as u32;
            }
        }
        out.push(best_idx);
    }
    Ok(out)
}

/// `np.hstack(probas)` — `k` `n_rows × width` blocks laid side by side.
///
/// The `voting='soft', flatten_transform=True` transform. Member-major, so
/// columns `j·width .. (j+1)·width` belong to member `j` — the layout
/// [`classifier_feature_names`] names.
pub fn hstack_blocks<F: Copy + Pod>(
    blocks: &[&[F]],
    n_rows: usize,
    width: usize,
) -> Result<Vec<F>> {
    check_columns(blocks, n_rows * width)?;
    if width == 0 {
        return Err(value_err(
            "a probability block must have at least one column",
        ));
    }
    let k = blocks.len();
    let stride = k * width;
    let mut out = vec![F::zeroed(); n_rows * stride];
    for (j, block) in blocks.iter().enumerate() {
        let offset = j * width;
        for r in 0..n_rows {
            out[r * stride + offset..r * stride + offset + width]
                .copy_from_slice(&block[r * width..(r + 1) * width]);
        }
    }
    Ok(out)
}

/// The label-column contract, shared by the host and device hard-vote arms.
///
/// Beyond [`check_columns`]' shape rules this enforces `label < n_bins`, which
/// the device kernel cannot check for itself: `counts[r · n_bins + label]` would
/// silently land in the next row's tally rather than fault.
fn check_label_columns(preds: &[&[u32]], n_rows: usize, n_bins: usize) -> Result<()> {
    check_columns(preds, n_rows)?;
    if n_bins == 0 {
        return Err(value_err("hard voting needs at least one class"));
    }
    for (j, pred) in preds.iter().enumerate() {
        if let Some(&bad) = pred.iter().find(|&&v| v as usize >= n_bins) {
            return Err(value_err(format!(
                "prediction column {j} holds label {bad}, which is not below \
                 the {n_bins} class slots"
            )));
        }
    }
    Ok(())
}

/// The `n_rows × k` transform matrix from `k` prediction columns.
///
/// sklearn's `_BaseVoting._predict` is `np.asarray([est.predict(X) for est in
/// self.estimators_]).T` — a `(k, n)` build followed by a transpose. The result
/// is materialised row-major here, which is the layout `weighted_average`
/// consumes and the one the Python shim reshapes to.
pub fn stack_columns<F: Copy + Pod>(preds: &[&[F]], n_rows: usize) -> Result<Vec<F>> {
    check_columns(preds, n_rows)?;
    let k = preds.len();
    let mut out = vec![F::zeroed(); n_rows * k];
    for (j, pred) in preds.iter().enumerate() {
        for r in 0..n_rows {
            out[r * k + j] = pred[r];
        }
    }
    Ok(out)
}

/// The weighted row mean of `k` prediction columns —
/// `np.average(mat, axis=1, weights=w)`.
///
/// `weights` is `None` for the uniform case, which numpy answers with
/// `mat.mean(axis=1)`; that is the same value as an all-ones weighting, because
/// `1.0 * x == x` exactly and the divisor is the same `k` either way, so the two
/// paths are folded into one here.
///
/// **The evaluation order is the contract.** numpy forms `mat * w` element-wise,
/// reduces along the row, and then DIVIDES by `w.sum()`. This function does the
/// same three things in the same order and in `F`'s precision — no `f64`
/// accumulator on an `f32` input, no reciprocal-multiply — because the oracle
/// compares the arms against numpy bit for bit.
///
/// The accumulation is LEFT TO RIGHT unconditionally. `np.add.reduce` blocks
/// pairwise above 8 elements when it can, which would reassociate the sum — but
/// the axis being reduced here is the `k` axis of an `(n, k)` array that numpy
/// itself built by transposing `(k, n)`, so it is strided rather than
/// contiguous. Measured rather than assumed: mlrs matches numpy EXACTLY at
/// `k = 1, 2, 5, 9, 16`, weighted and uniform, on both sides of that threshold
/// (`test_many_members_still_match_numpys_reduction_exactly`). Anything that
/// changes the shape handed to numpy should re-check that test rather than trust
/// this paragraph.
///
/// A zero weight sum is [`VotingError::ZeroWeightSum`], which is what numpy
/// raises rather than returning infinities.
pub fn weighted_average<F>(preds: &[&[F]], weights: Option<&[F]>, n_rows: usize) -> Result<Vec<F>>
where
    F: Copy + Pod + PartialEq + Add<Output = F> + Mul<Output = F> + Div<Output = F>,
{
    check_columns(preds, n_rows)?;
    let k = preds.len();
    let (w, denom) = resolve_weights::<F>(weights, k)?;

    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        // Seeded from the first member rather than from `F::default()`: `0 + x`
        // is not `x` when `x` is `-0.0`, and the sum is compared against numpy's
        // exactly.
        let mut acc = preds[0][r] * w[0];
        for j in 1..k {
            acc = acc + preds[j][r] * w[j];
        }
        out.push(acc / denom);
    }
    Ok(out)
}

/// The per-member weights and their sum, with `None` expanded to all-ones.
///
/// The sum is accumulated left to right in `F`, matching `w.sum()`; it is
/// computed ONCE here and handed to both the host loop and the device kernel, so
/// the two arms divide by identical bits rather than by two independently
/// rounded sums.
fn resolve_weights<F>(weights: Option<&[F]>, k: usize) -> Result<(Vec<F>, F)>
where
    F: Copy + Pod + PartialEq + Add<Output = F>,
{
    let w: Vec<F> = match weights {
        Some(w) => {
            if w.len() != k {
                return Err(value_err(format!(
                    "have {k} prediction columns but {} weights",
                    w.len()
                )));
            }
            w.to_vec()
        }
        None => vec![one::<F>(); k],
    };
    let mut denom = w[0];
    for &wj in &w[1..] {
        denom = denom + wj;
    }
    if denom == F::zeroed() {
        return Err(VotingError::ZeroWeightSum);
    }
    Ok((w, denom))
}

/// `1.0` in `F`, for the uniform-weight expansion.
///
/// `F` here is only ever `f32`/`f64` (the two float types every mlrs kernel is
/// generic over), but this module's bounds are the arithmetic ones rather than
/// cubecl's `Float`, so the constant is produced by the same `size_of`
/// bytemuck dispatch `mlrs_backend::prims::covariance::recip` uses.
fn one<F: Pod>() -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&1.0f32)),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&1.0f64)),
        _ => unreachable!("voting aggregation is f32/f64 only"),
    }
}

/// The shape contract every arm holds callers to.
///
/// Factored out so the host and device arms reject identically: a kernel cannot
/// return an error, so the device route has to be told "no" here or not at all,
/// and a second validator would report a different message for the same mistake.
fn check_columns<F>(preds: &[&[F]], n_rows: usize) -> Result<()> {
    if preds.is_empty() {
        return Err(value_err(
            "All estimators are dropped. At least one is required to be an estimator.",
        ));
    }
    for (j, pred) in preds.iter().enumerate() {
        if pred.len() != n_rows {
            return Err(value_err(format!(
                "prediction column {j} has {} elements, expected n_rows = {n_rows}",
                pred.len()
            )));
        }
    }
    Ok(())
}

/// Stack the prediction columns on the arm `engine` names (VOTE-01).
///
/// [`VoteEngine::Numpy`] never reaches here — it is the shim's own
/// `np.asarray(...).T` and is resolved before the FFI boundary is crossed — so
/// it is treated as `Host`, which is what a Rust-native caller asking for "not
/// the device" means.
///
/// Both arms produce the same bytes: the scatter performs no arithmetic and
/// writes each column to the offsets this host loop writes them to
/// (`crates/mlrs-backend/tests/voting_test.rs` asserts exactly that).
pub fn vote_transform<F>(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    preds: &[&[F]],
    n_rows: usize,
) -> Result<Vec<F>>
where
    F: Float + CubeElement + Pod,
{
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => stack_columns(preds, n_rows),
        VoteEngine::Device => {
            // The host arm's validation runs FIRST even on the device route, so
            // a mis-shaped column reports the same sklearn-shaped message on
            // both arms instead of a `PrimError` on one of them.
            check_columns(preds, n_rows)?;
            vote_transform_device(pool, preds, n_rows).map_err(|e| value_err(e.to_string()))
        }
    }
}

/// Average the prediction columns on the arm `engine` names (VOTE-01).
///
/// Unlike [`vote_transform`] this one COMPUTES, and that costs the exact
/// equality: the kernel accumulates in the same member order and divides by the
/// SAME host-computed `denom`, but a GPU backend contracts `acc + pred·w` into a
/// fused multiply-add and so rounds ONCE where this loop (and numpy) round
/// twice. Measured on rocm gfx1151 at f32 the two arms agree to within one ULP;
/// on the cpu backend, which does not contract, they are bit-identical.
///
/// So the backend test holds the device arm to a few ULP rather than to
/// equality, and the host arm — which is what `MLRS_VOTING_ENGINE=host` and the
/// default `numpy` arm both produce — stays bit-identical to `np.average`. See
/// `mlrs_kernels::voting` for why the contraction is not suppressible.
pub fn vote_average<F>(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    preds: &[&[F]],
    weights: Option<&[F]>,
    n_rows: usize,
) -> Result<Vec<F>>
where
    F: Float + CubeElement + Pod,
{
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => weighted_average(preds, weights, n_rows),
        VoteEngine::Device => {
            check_columns(preds, n_rows)?;
            let (w, denom) = resolve_weights::<F>(weights, preds.len())?;
            vote_average_device(pool, preds, &w, denom, n_rows)
                .map_err(|e| value_err(e.to_string()))
        }
    }
}

/// Weighted majority voting on the arm `engine` names (VOTE-CLF-01).
///
/// Both arms tally in `f64` — see [`hard_vote_labels`] for why that is numpy's
/// width and not a choice — so unlike [`vote_average`] the two agree **exactly**
/// here rather than to a ULP. There is no `acc + a·b` to contract: the kernel
/// adds a scalar weight to a bin, which is one rounding on every backend.
///
/// A tie is still a tie in both arms, and both break it at the lowest index, so
/// the LABELS agree and not merely the counts.
pub fn vote_hard_predict(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    preds: &[&[u32]],
    weights: Option<&[f64]>,
    n_rows: usize,
    n_bins: usize,
) -> Result<Vec<u32>> {
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => hard_vote_labels(preds, weights, n_rows, n_bins),
        VoteEngine::Device => {
            // The host arm's validation runs FIRST on the device route too, so a
            // mis-shaped or out-of-range column reports the same sklearn-shaped
            // message on both arms instead of a `PrimError` on one of them.
            check_label_columns(preds, n_rows, n_bins)?;
            let w: Vec<f64> = match weights {
                Some(w) if w.len() != preds.len() => {
                    return Err(value_err(format!(
                        "have {} prediction columns but {} weights",
                        preds.len(),
                        w.len()
                    )))
                }
                Some(w) => w.to_vec(),
                None => vec![1.0; preds.len()],
            };
            vote_hard_predict_device(pool, preds, &w, n_rows, n_bins as u32)
                .map_err(|e| value_err(e.to_string()))
        }
    }
}

/// `predict_proba` under `voting='soft'`, on the arm `engine` names
/// (VOTE-CLF-01).
///
/// `np.average(probas, axis=0, weights=w)` over `k` `n_rows × n_cols` blocks —
/// which is [`vote_average`]'s arithmetic with `n_rows · n_cols` in place of
/// `n_rows`, and is implemented as exactly that rather than as a second
/// reduction. The one-ULP device caveat [`vote_average`] documents applies here
/// unchanged, for the same FMA-contraction reason.
pub fn vote_soft_proba<F>(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    weights: Option<&[F]>,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<F>>
where
    F: Float + CubeElement + Pod,
{
    let len = n_rows * n_cols;
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => weighted_average(blocks, weights, len),
        VoteEngine::Device => {
            check_columns(blocks, len)?;
            let (w, denom) = resolve_weights::<F>(weights, blocks.len())?;
            vote_soft_proba_device(pool, blocks, &w, denom, n_rows, n_cols)
                .map_err(|e| value_err(e.to_string()))
        }
    }
}

/// `predict` under `voting='soft'`, on the arm `engine` names (VOTE-CLF-01).
///
/// `argmax(np.average(probas, axis=0, weights=w), axis=1)`. The host arm runs
/// [`vote_soft_proba`] and then [`argmax_rows`]; the device arm FUSES the two so
/// the `n_rows × n_cols` average never crosses the bus, which is the one place
/// in this module where the device has a structural rather than a hoped-for
/// advantage.
///
/// The label a row gets is the same on both arms unless the average's top two
/// classes are within the device's one-ULP contraction gap of each other — which
/// is exactly the case where the two answers are equally defensible and numpy's
/// own tie-break is already arbitrary. The oracle tests use well-separated
/// probabilities for that reason and assert labels, not merely accuracy.
pub fn vote_soft_predict<F>(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    weights: Option<&[F]>,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<u32>>
where
    F: Float + CubeElement + Pod,
{
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => {
            let avg = weighted_average(blocks, weights, n_rows * n_cols)?;
            argmax_rows(&avg, n_rows, n_cols)
        }
        VoteEngine::Device => {
            check_columns(blocks, n_rows * n_cols)?;
            let (w, denom) = resolve_weights::<F>(weights, blocks.len())?;
            vote_soft_predict_device(pool, blocks, &w, denom, n_rows, n_cols)
                .map_err(|e| value_err(e.to_string()))
        }
    }
}

/// `transform` under `voting='soft', flatten_transform=True`, on the arm
/// `engine` names (VOTE-CLF-01).
///
/// `np.hstack(probas)`. Like [`vote_transform`] this performs no arithmetic, so
/// all three arms are byte-identical; and like [`vote_transform`] it is the
/// copy-shaped half whose measured place is `docs/voting.md`'s to settle.
pub fn vote_hstack<F>(
    engine: VoteEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    n_rows: usize,
    width: usize,
) -> Result<Vec<F>>
where
    F: Float + CubeElement + Pod,
{
    match engine {
        VoteEngine::Numpy | VoteEngine::Host => hstack_blocks(blocks, n_rows, width),
        VoteEngine::Device => {
            check_columns(blocks, n_rows * width)?;
            vote_hstack_device(pool, blocks, n_rows, width).map_err(|e| value_err(e.to_string()))
        }
    }
}
