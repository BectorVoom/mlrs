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
use mlrs_backend::prims::voting::{vote_average_device, vote_transform_device, VoteEngine};
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
